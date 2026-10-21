"""Independent term and CPU reference checks for assembled component Systems.

The reference is rebuilt from separately retained, declarative force-field
resources. Assembly XML is reopened only after its byte bindings are checked.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import xml.etree.ElementTree as etree
from decimal import Decimal, InvalidOperation

import numpy as np
import rfc8785
from jsonschema import Draft202012Validator

from neomd.contracts import schema_dir
from neomd_preparation.components.assets import validate_package_assets
from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    validate_document,
)

_MAX_CONTROL = 1_048_576
_MAX_ASSEMBLY_JSON = 268_435_456
_MAX_ASSEMBLY_REPORT = 16_777_216
_MAX_SYSTEM_XML = 268_435_456
_MAX_PARTICLES = 2_000_000
_TERM_ABS = 1e-9
_TERM_REL = 1e-11
_BOX_ABS = 1e-12
_ENERGY_ABS = 1e-3
_ENERGY_REL = 1e-7
_FORCE_ABS = 1e-2
_FORCE_REL = 1e-6
_FORCE_CLASSES = {"HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce",
                  "NonbondedForce"}
_XML_TOP = {"Info", "AtomTypes", "Residues", *_FORCE_CLASSES}
_XML_FORBIDDEN = {"Include", "Script", "InitializationScript", "VirtualSite"}
_RESOURCE_KEYS = ("protein", "water_ions", "gaff", "ligand_template")
_RESOURCE_PROTEIN_KEYS = _RESOURCE_KEYS[:2]
_RESOURCE_OPENFF_KEYS = ("protein", "water_ions", "ligand_template")
_RESOURCE_SEP_KEYS = ("protein", "phosaa", "water_ions")
_QC_SCHEMA_DIR = schema_dir("component-system-qc/v1")
_QC_VALIDATOR = Draft202012Validator(json.loads((_QC_SCHEMA_DIR / "report.schema.json").read_text()))


def _fail(path, detail, code="incompatible_package"):
    raise ComponentContractError(code, path, detail)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data, path, *, limit=_MAX_CONTROL):
    if type(data) is not bytes or not 0 < len(data) <= limit:
        _fail(path, "JSON exceeds its byte limit or is not exact bytes", "oversized_payload")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _fail(path, "duplicate JSON field", "invalid_reference")
            value[key] = item
        return value

    try:
        document = json.loads(data, object_pairs_hook=unique,
                              parse_constant=lambda _: _fail(path, "nonfinite JSON number",
                                                              "invalid_reference"))
        if rfc8785.dumps(document) != data:
            _fail(path, "expected RFC 8785 canonical JSON", "invalid_reference")
    except ComponentContractError:
        raise
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError):
        _fail(path, "invalid canonical JSON", "invalid_reference")
    return document


def _safe_forcefield_xml(data, role):
    path = f"/reference_resources/{role}"
    if type(data) is not bytes or not 0 < len(data) <= 32 * 1_048_576:
        _fail(path, "reference XML exceeds 32 MiB or is not exact bytes", "oversized_payload")
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        _fail(path, "XML declarations and entities are unsupported", "invalid_reference")
    try:
        root = etree.fromstring(data)
    except etree.ParseError:
        _fail(path, "invalid force-field XML", "invalid_reference")
    if (root.tag != "ForceField" or any(child.tag not in _XML_TOP for child in root)
            or any(node.tag in _XML_FORBIDDEN for node in root.iter())):
        _fail(path, "reference XML contains unsupported executable or included elements",
              "invalid_reference")
    return data


def build_direct_forcefield_reference(topology, resources):
    """Construct the fixed ff14SB/TIP3P/ligand reference without assembly code.

    ``resources`` has protein and water XML bytes, plus GAFF and template XML
    bytes when a ligand is assigned. The QC entrypoint binds their hashes to
    independently retained package/resource evidence.
    """
    import openmm as mm
    from openmm import app, unit

    if (not isinstance(resources, dict)
            or set(resources) not in (set(_RESOURCE_PROTEIN_KEYS), set(_RESOURCE_KEYS),
                                     set(_RESOURCE_OPENFF_KEYS), set(_RESOURCE_SEP_KEYS))):
        _fail("/reference_resources", "qualified reference XML roles are required",
              "invalid_reference")
    roles = (_RESOURCE_KEYS if "gaff" in resources else _RESOURCE_OPENFF_KEYS
             if "ligand_template" in resources else _RESOURCE_SEP_KEYS
             if "phosaa" in resources else _RESOURCE_PROTEIN_KEYS)
    chunks = [_safe_forcefield_xml(resources[role], role) for role in roles]
    field = app.ForceField(*(io.BytesIO(data) for data in chunks))
    try:
        system = field.createSystem(
            topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds, rigidWater=True, hydrogenMass=4.0 * unit.amu,
            removeCMMotion=False, switchDistance=None, flexibleConstraints=False,
            ewaldErrorTolerance=0.0005, useDispersionCorrection=True,
            residueTemplates={}, ignoreExternalBonds=False,
        )
    except (ValueError, RuntimeError, TypeError) as error:
        _fail("/reference_resources", f"direct ForceField reference failed: {type(error).__name__}",
              "capability_unavailable")
    nonbonded = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
    if len(nonbonded) != 1:
        _fail("/reference_resources", "direct reference requires one NonbondedForce")
    nonbonded[0].setExceptionsUsePeriodicBoundaryConditions(False)
    return system, {role: _hash(resources[role]) for role in roles}


def _value(quantity, target):
    return float(quantity.value_in_unit(target))


def _ordered_pair(a, b):
    return [min(int(a), int(b)), max(int(a), int(b))]


def snapshot_system_terms(system):
    """Read all supported OpenMM terms independently of package extraction."""
    import openmm as mm
    from openmm import unit

    count = system.getNumParticles()
    if not 0 < count <= _MAX_PARTICLES:
        _fail("/system/particles", "unsupported particle count", "oversized_payload")
    if any(system.isVirtualSite(index) for index in range(count)):
        _fail("/system/virtual_sites", "first profile forbids virtual sites",
              "capability_unavailable")
    forces = list(system.getForces())
    kinds = [type(force).__name__ for force in forces]
    if len(kinds) != len(set(kinds)) or set(kinds) != _FORCE_CLASSES:
        _fail("/system/forces", "expected exactly four qualified force classes")
    by_type = {type(force).__name__: force for force in forces}
    nonbonded = by_type["NonbondedForce"]
    if nonbonded.getNonbondedMethod() != mm.NonbondedForce.PME:
        _fail("/system/nonbonded/method", "first profile requires PME")
    if (nonbonded.getNumGlobalParameters() or nonbonded.getNumParticleParameterOffsets()
            or nonbonded.getNumExceptionParameterOffsets()):
        _fail("/system/nonbonded/offsets", "parameter offsets require another profile",
              "capability_unavailable")
    if nonbonded.getNumParticles() != count:
        _fail("/system/nonbonded/particles", "particle parameter coverage differs")
    masses = [_value(system.getParticleMass(i), unit.dalton) for i in range(count)]
    particles = []
    for index in range(count):
        charge, sigma, epsilon = nonbonded.getParticleParameters(index)
        particles.append({"mass_amu": masses[index],
                          "charge_e": _value(charge, unit.elementary_charge),
                          "sigma_nm": _value(sigma, unit.nanometer),
                          "epsilon_kj_mol": _value(epsilon, unit.kilojoule_per_mole)})
    bonds = []
    force = by_type["HarmonicBondForce"]
    for index in range(force.getNumBonds()):
        a, b, length, stiffness = force.getBondParameters(index)
        bonds.append({"particle_indices": _ordered_pair(a, b),
                      "length_nm": _value(length, unit.nanometer),
                      "k_kj_mol_nm2": _value(stiffness, unit.kilojoule_per_mole / unit.nanometer**2)})
    angles = []
    force = by_type["HarmonicAngleForce"]
    for index in range(force.getNumAngles()):
        a, b, c, angle, stiffness = force.getAngleParameters(index)
        angles.append({"particle_indices": [int(a), int(b), int(c)],
                       "angle_rad": _value(angle, unit.radian),
                       "k_kj_mol_rad2": _value(stiffness, unit.kilojoule_per_mole / unit.radian**2)})
    torsions = []
    force = by_type["PeriodicTorsionForce"]
    for index in range(force.getNumTorsions()):
        a, b, c, d, periodicity, phase, stiffness = force.getTorsionParameters(index)
        torsions.append({"particle_indices": [int(a), int(b), int(c), int(d)],
                         "periodicity": int(periodicity), "phase_rad": _value(phase, unit.radian),
                         "k_kj_mol": _value(stiffness, unit.kilojoule_per_mole)})
    exceptions = []
    for index in range(nonbonded.getNumExceptions()):
        a, b, charge_product, sigma, epsilon = nonbonded.getExceptionParameters(index)
        exceptions.append({"particle_indices": _ordered_pair(a, b),
                           "chargeprod_e2": _value(charge_product, unit.elementary_charge**2),
                           "sigma_nm": _value(sigma, unit.nanometer),
                           "epsilon_kj_mol": _value(epsilon, unit.kilojoule_per_mole)})
    constraints = []
    for index in range(system.getNumConstraints()):
        a, b, distance = system.getConstraintParameters(index)
        constraints.append({"particle_indices": _ordered_pair(a, b),
                            "distance_nm": _value(distance, unit.nanometer)})
    boxes = [[_value(component, unit.nanometer) for component in vector]
             for vector in system.getDefaultPeriodicBoxVectors()]
    result = {
        "particle_count": count,
        "force_types": sorted(kinds),
        "force_groups": {kind: by_type[kind].getForceGroup() for kind in sorted(kinds)},
        "force_periodic": {kind: by_type[kind].usesPeriodicBoundaryConditions()
                           for kind in sorted(kinds)},
        "particles": particles,
        "bonds": sorted(bonds, key=lambda row: row["particle_indices"]),
        "angles": sorted(angles, key=lambda row: row["particle_indices"]),
        "torsions": sorted(torsions, key=lambda row: (row["particle_indices"],
                                                  row["periodicity"], row["phase_rad"])),
        "exceptions": sorted(exceptions, key=lambda row: row["particle_indices"]),
        "constraints": sorted(constraints, key=lambda row: row["particle_indices"]),
        "box_vectors_nm": boxes,
        "nonbonded": {
            "method": "pme",
            "cutoff_nm": _value(nonbonded.getCutoffDistance(), unit.nanometer),
            "ewald_error_tolerance": nonbonded.getEwaldErrorTolerance(),
            "use_dispersion_correction": nonbonded.getUseDispersionCorrection(),
            "use_switching_function": nonbonded.getUseSwitchingFunction(),
            "switching_distance_nm": _value(nonbonded.getSwitchingDistance(), unit.nanometer),
            "exceptions_use_periodic_boundary_conditions":
                nonbonded.getExceptionsUsePeriodicBoundaryConditions(),
            "pme_parameters": [(_value(value, unit.nanometer**-1) if index == 0 else int(value))
                               for index, value in enumerate(nonbonded.getPMEParameters())],
            "reciprocal_space_force_group": nonbonded.getReciprocalSpaceForceGroup(),
        },
    }
    if not system.usesPeriodicBoundaryConditions():
        _fail("/system/box_vectors_nm", "first profile requires a periodic System")
    return result


def _compare(actual, expected, path, summary, *, box=False):
    if type(actual) is not type(expected) and not (
            type(actual) in (int, float) and type(expected) in (int, float)):
        _fail(path, "reference term type differs")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            _fail(path, "reference term fields differ")
        for key in expected:
            _compare(actual[key], expected[key], f"{path}/{key}", summary,
                     box=box or key == "box_vectors_nm")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            _fail(path, "reference term count differs")
        for index, (value, target) in enumerate(zip(actual, expected)):
            _compare(value, target, f"{path}/{index}", summary, box=box)
    elif type(expected) in (float, int):
        if not math.isfinite(actual) or not math.isfinite(expected):
            _fail(path, "term is nonfinite")
        difference = abs(actual - expected)
        tolerance = _BOX_ABS if box else _TERM_ABS + _TERM_REL * abs(expected)
        if difference > tolerance:
            _fail(path, f"reference term differs by {difference:.8g} (limit {tolerance:.8g})")
        summary["max_term_abs_delta"] = max(summary["max_term_abs_delta"], difference)
    elif actual != expected:
        _fail(path, "reference term differs")


def compare_system_terms(subject, reference):
    """Compare every supported force term and return the maximum absolute delta."""
    summary = {"max_term_abs_delta": 0.0}
    _compare(snapshot_system_terms(subject), snapshot_system_terms(reference),
             "/system", summary)
    return summary


def compare_package_force_subset(system, force_asset, state_particle_indices, *, net_formal_charge):
    """Compare all source package terms for one noncovalent instance subset.

    ``state_particle_indices`` is the independently verified instance map from
    the assembly manifest. Cross-boundary bonded terms need a separately
    versioned junction reference and therefore reject in this first profile.
    """
    if (not isinstance(force_asset, dict) or force_asset.get("format") !=
            "neomd-component-parameters-v1" or not isinstance(state_particle_indices, dict)
            or not state_particle_indices):
        _fail("/package_force", "complete source package force terms and map are required",
              "invalid_reference")
    subject = snapshot_system_terms(system)
    count = subject["particle_count"]
    if (any(type(key) is not str or type(value) is not int or not 0 <= value < count
            for key, value in state_particle_indices.items())
            or len(set(state_particle_indices.values())) != len(state_particle_indices)):
        _fail("/package_force/map", "state particles must map one-to-one to System indices",
              "mapping_mismatch")
    selected = set(state_particle_indices.values())
    rows = force_asset.get("atoms")
    if (not isinstance(rows, list) or len(rows) != len(selected)
            or {row.get("state_atom_id") for row in rows if isinstance(row, dict)}
            != set(state_particle_indices)):
        _fail("/package_force/atoms", "force atom map is incomplete", "mapping_mismatch")
    total = Decimal(0)
    summary = {"max_term_abs_delta": 0.0}
    for row in rows:
        sid = row["state_atom_id"]
        index = state_particle_indices[sid]
        try:
            total += Decimal(row["charge_e"])
        except (KeyError, TypeError, ValueError, InvalidOperation):
            _fail(f"/package_force/atoms/{sid}", "exact source charge is invalid",
                  "invalid_reference")
        expected = {key: row[key] for key in ("mass_amu", "sigma_nm", "epsilon_kj_mol")}
        expected["charge_e"] = float(row["charge_e"])
        _compare(subject["particles"][index], expected,
                 f"/package_force/atoms/{sid}", summary)
    residual = total - Decimal(net_formal_charge)
    if not total.is_finite() or abs(residual) > Decimal("0.000001"):
        _fail("/package_force/charge_sum_e", "source charge sum differs from reviewed state")

    def mapped(row, source):
        ids = row.get("atom_ids")
        if not isinstance(ids, list) or any(sid not in state_particle_indices for sid in ids):
            _fail(f"/package_force/{source}", "unknown term atom ID", "mapping_mismatch")
        result = {key: value for key, value in row.items() if key not in ("atom_ids", "kind")}
        indices = [state_particle_indices[sid] for sid in ids]
        result["particle_indices"] = _ordered_pair(*indices) if len(indices) == 2 else indices
        return result

    for source, target in (("bonds", "bonds"), ("angles", "angles"),
                           ("torsions", "torsions"), ("exceptions", "exceptions"),
                           ("constraints", "constraints")):
        expected_rows = force_asset.get(source)
        if not isinstance(expected_rows, list):
            _fail(f"/package_force/{source}", "source force terms are missing",
                  "invalid_reference")
        expected = [mapped(row, source) for row in expected_rows]
        actual = []
        for row in subject[target]:
            overlap = selected.intersection(row["particle_indices"])
            if overlap and len(overlap) != len(row["particle_indices"]):
                _fail(f"/package_force/{source}", "unreferenced cross-component term")
            if overlap:
                actual.append(row)
        order = (lambda row: (row["particle_indices"], row["periodicity"], row["phase_rad"]))\
            if source == "torsions" else (lambda row: row["particle_indices"])
        _compare(sorted(actual, key=order), sorted(expected, key=order),
                 f"/package_force/{source}", summary)
    if force_asset.get("virtual_sites") != []:
        _fail("/package_force/virtual_sites", "virtual sites require another qualified profile",
              "capability_unavailable")
    return {"charge_sum_e": str(total), "charge_residual_e": str(residual), **summary}


def _cpu_observables(system, positions_nm):
    import openmm as mm
    from openmm import unit

    integrator = mm.VerletIntegrator(0.001 * unit.picosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"),
                         {"Threads": "1"})
    try:
        context.setPositions(positions_nm * unit.nanometer)
        state = context.getState(getEnergy=True, getForces=True)
        energy = _value(state.getPotentialEnergy(), unit.kilojoule_per_mole)
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer), dtype=np.float64)
        if not math.isfinite(energy) or forces.shape != positions_nm.shape or not np.isfinite(forces).all():
            _fail("/system/observables", "CPU energy or forces are nonfinite")
        return energy, forces
    finally:
        del context
        del integrator


def compare_cpu_observables(subject, reference, positions_nm):
    """Evaluate both Systems on CPU with identical coordinates and no steps."""
    positions = np.asarray(positions_nm, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape != (subject.getNumParticles(), 3)
            or reference.getNumParticles() != subject.getNumParticles()
            or not np.isfinite(positions).all()):
        _fail("/positions", "finite N×3 positions must match both Systems", "mapping_mismatch")
    energy, forces = _cpu_observables(subject, positions)
    expected_energy, expected_forces = _cpu_observables(reference, positions)
    energy_delta = abs(energy - expected_energy)
    energy_limit = _ENERGY_ABS + _ENERGY_REL * abs(expected_energy)
    if energy_delta > energy_limit:
        _fail("/system/energy", f"CPU reference energy differs by {energy_delta:.8g}")
    differences = np.abs(forces - expected_forces)
    limits = _FORCE_ABS + _FORCE_REL * np.abs(expected_forces)
    if not np.all(differences <= limits):
        index = np.unravel_index(np.argmax(differences - limits), differences.shape)
        _fail(f"/system/forces/{index[0]}/{index[1]}",
              f"CPU reference force differs by {differences[index]:.8g}")
    return {"subject_energy_kj_mol": energy,
            "reference_energy_kj_mol": expected_energy,
            "energy_abs_delta_kj_mol": energy_delta,
            "max_force_abs_delta_kj_mol_nm": float(np.max(differences, initial=0.0))}


def _bound_assembly_bytes(assembly, expected_manifest_sha256):
    """Verify transfer bytes and row counts before System XML deserialization."""
    fields = {"topology_json": "topology_bytes", "positions_f64le": "positions_f64le",
              "system_xml": "system_xml", "report_json": "report_bytes"}
    raw_manifest = getattr(assembly, "manifest_bytes", None)
    manifest = _canonical_json(raw_manifest, "/manifest", limit=_MAX_ASSEMBLY_JSON)
    if expected_manifest_sha256 is not None and _hash(raw_manifest) != expected_manifest_sha256:
        _fail("/manifest", "assembly manifest differs from trusted hash", "invalid_reference")
    if (not isinstance(manifest, dict) or manifest.get("format") != "neomd-component-assembly-v1"
            or set(manifest) != {"format", "recipe_sha256", "source_content_sha256",
                                 "box_policy", "source_box_vectors_nm", "box_vectors_nm",
                                 "particles", "pose_receipts_sha256", "artifacts"}):
        _fail("/manifest", "unsupported assembly manifest", "contract_version_mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(fields):
        _fail("/manifest/artifacts", "four exact output artifact bindings are required",
              "invalid_reference")
    payloads = {}
    limits = {"topology_json": _MAX_ASSEMBLY_JSON,
              "positions_f64le": 24 * _MAX_PARTICLES,
              "system_xml": _MAX_SYSTEM_XML,
              "report_json": _MAX_ASSEMBLY_REPORT}
    for name, attribute in fields.items():
        data = getattr(assembly, attribute, None)
        descriptor = artifacts[name]
        if type(data) is not bytes or len(data) > limits[name]:
            _fail(f"/manifest/artifacts/{name}", "assembled artifact exceeds byte limit",
                  "oversized_payload")
        if (not isinstance(descriptor, dict)
                or set(descriptor) != {"sha256", "size_bytes"}
                or type(descriptor["size_bytes"]) is not int
                or descriptor["size_bytes"] != len(data)
                or type(descriptor["sha256"]) is not str
                or descriptor["sha256"] != _hash(data)):
            _fail(f"/manifest/artifacts/{name}", "exact encoded artifact identity differs",
                  "invalid_reference")
        payloads[name] = data
    topology = _canonical_json(payloads["topology_json"], "/topology",
                               limit=_MAX_ASSEMBLY_JSON)
    report = _canonical_json(payloads["report_json"], "/assembly_report",
                             limit=_MAX_ASSEMBLY_REPORT)
    rows = manifest["particles"]
    if not isinstance(rows, list) or not 0 < len(rows) <= _MAX_PARTICLES:
        _fail("/manifest/particles", "bounded particle map is required", "oversized_payload")
    if len(payloads["positions_f64le"]) != 24 * len(rows):
        _fail("/manifest/artifacts/positions_f64le", "N×3 float64 coordinates differ",
              "mapping_mismatch")
    positions = np.frombuffer(payloads["positions_f64le"], dtype="<f8").reshape((-1, 3))
    if not np.isfinite(positions).all():
        _fail("/positions", "coordinates must be finite", "mapping_mismatch")
    return manifest, topology, report, positions, payloads["system_xml"]


def topology_from_document(document, manifest):
    """Recreate the exact graph and chain context through the public app API."""
    from openmm import Vec3, app, unit

    if (not isinstance(document, dict) or set(document) !=
            {"format", "atoms", "residues", "bonds", "box_vectors_nm"}
            or document["format"] != "neomd-component-assembled-topology-v1"):
        _fail("/topology", "unsupported assembled topology", "contract_version_mismatch")
    atoms, residues, bonds = (document[key] for key in ("atoms", "residues", "bonds"))
    particles = manifest["particles"]
    if (not all(isinstance(rows, list) for rows in (atoms, residues, bonds))
            or len(atoms) != len(particles)):
        _fail("/topology/atoms", "topology and particle map coverage differs",
              "mapping_mismatch")
    topology = app.Topology()
    handles = []
    chains = {}
    visited = set()
    previous_chain = -1
    for residue_index, row in enumerate(residues):
        if (not isinstance(row, dict) or set(row) !=
                {"residue_index", "chain_index", "name", "atom_indices"}
                or type(row["residue_index"]) is not int
                or row["residue_index"] != residue_index
                or type(row["chain_index"]) is not int or row["chain_index"] < 0
                or type(row["name"]) is not str or not row["name"]
                or not isinstance(row["atom_indices"], list) or not row["atom_indices"]):
            _fail(f"/topology/residues/{residue_index}", "invalid residue/chain row",
                  "mapping_mismatch")
        chain_index = row["chain_index"]
        if chain_index < previous_chain:
            _fail(f"/topology/residues/{residue_index}/chain_index",
                  "chain order differs from particle order", "mapping_mismatch")
        previous_chain = chain_index
        if chain_index not in chains:
            if chain_index != len(chains):
                _fail(f"/topology/residues/{residue_index}/chain_index",
                      "chain indices must be contiguous", "mapping_mismatch")
            chains[chain_index] = topology.addChain()
        residue = topology.addResidue(row["name"], chains[chain_index])
        for particle_index in row["atom_indices"]:
            if (type(particle_index) is not int or particle_index != len(handles)
                    or particle_index in visited or particle_index >= len(atoms)):
                _fail(f"/topology/residues/{residue_index}/atom_indices",
                      "residue atom order differs from particle order", "mapping_mismatch")
            visited.add(particle_index)
            atom = atoms[particle_index]
            particle = particles[particle_index]
            if (not isinstance(atom, dict) or set(atom) !=
                    {"particle_index", "residue_index", "name", "element"}
                    or atom["particle_index"] != particle_index
                    or atom["residue_index"] != residue_index
                    or particle.get("particle_index") != particle_index
                    or particle.get("residue_index") != residue_index
                    or atom["name"] != particle.get("name")
                    or atom["element"] != particle.get("element")):
                _fail(f"/topology/atoms/{particle_index}",
                      "topology atom differs from assembly particle map", "mapping_mismatch")
            try:
                element = app.element.get_by_symbol(atom["element"])
            except (KeyError, TypeError, ValueError):
                _fail(f"/topology/atoms/{particle_index}/element",
                      "unrecognized atom element", "mapping_mismatch")
            handles.append(topology.addAtom(atom["name"], element, residue))
    if len(handles) != len(atoms):
        _fail("/topology/atoms", "residue membership is incomplete", "mapping_mismatch")
    previous = None
    for index, pair in enumerate(bonds):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or not 0 <= pair[0] < pair[1] < len(handles)
                or (previous is not None and pair <= previous)):
            _fail(f"/topology/bonds/{index}", "bond pair is invalid or duplicate",
                  "mapping_mismatch")
        topology.addBond(handles[pair[0]], handles[pair[1]])
        previous = pair
    box = document["box_vectors_nm"]
    if (not isinstance(box, list) or len(box) != 3
            or any(not isinstance(row, list) or len(row) != 3
                   or any(type(value) not in (int, float) or not math.isfinite(value)
                          for value in row) for row in box)):
        _fail("/topology/box_vectors_nm", "finite 3×3 box required", "mapping_mismatch")
    topology.setPeriodicBoxVectors([Vec3(*row) for row in box] * unit.nanometer)
    summary = {"max_term_abs_delta": 0.0}
    _compare(box, manifest["box_vectors_nm"], "/manifest/box_vectors_nm", summary, box=True)
    return topology



# Backward-compatible private spelling; the public name is topology_from_document.
_topology_from_document = topology_from_document


def _deserialize_bounded_system(system_xml):
    import openmm as mm

    if b"<!DOCTYPE" in system_xml or b"<!ENTITY" in system_xml:
        _fail("/system_xml", "XML declarations/entities are unsupported", "invalid_reference")
    try:
        root = etree.fromstring(system_xml)
    except etree.ParseError:
        _fail("/system_xml", "System XML is malformed", "invalid_reference")
    forces = root.find("Forces")
    force_kinds = [] if forces is None else [child.get("type") for child in forces]
    if (root.tag != "System" or root.get("type") != "System" or forces is None
            or any(type(kind) is not str for kind in force_kinds)
            or sorted(force_kinds) != sorted(_FORCE_CLASSES)):
        _fail("/system_xml", "System XML force vocabulary is unsupported", "invalid_reference")
    try:
        return mm.XmlSerializer.deserialize(system_xml.decode("utf-8"))
    except (RuntimeError, UnicodeDecodeError, ValueError):
        _fail("/system_xml", "System XML cannot be reopened", "invalid_reference")


def _bind_scientific_inputs(manifest, assembly_report, recipe, states, instances, packages,
                            assets_by_package):
    validate_document("preparation-recipe", recipe)
    recipe_sha = canonical_sha256(recipe)
    source_sha = recipe["source_topology"]["content_sha256"]
    if (manifest["recipe_sha256"] != recipe_sha
            or manifest["source_content_sha256"] != source_sha
            or manifest["box_policy"] != recipe["assembly"]["box_policy"]):
        _fail("/manifest", "assembly is bound to another immutable recipe/source",
              "mapping_mismatch")
    if (not isinstance(assembly_report, dict)
            or assembly_report.get("format") != "neomd-component-assembly-report-v1"
            or assembly_report.get("qc_status") != "assembled_pending_independent_qc"
            or assembly_report.get("recipe_sha256") != recipe_sha
            or assembly_report.get("source_content_sha256") != source_sha
            or assembly_report.get("assembly") != recipe["assembly"]
            or assembly_report.get("box_vectors_nm") != manifest["box_vectors_nm"]
            or assembly_report.get("source_box_vectors_nm") != manifest["source_box_vectors_nm"]
            or assembly_report.get("pose_receipts_sha256") != manifest["pose_receipts_sha256"]):
        _fail("/assembly_report", "assembly receipt differs from immutable inputs",
              "mapping_mismatch")
    by_state = {}
    for state in states:
        validate_document("chemical-state", state)
        if state["state_id"] in by_state:
            _fail("/states", "duplicate state ID", "mapping_mismatch")
        by_state[state["state_id"]] = state
    by_instance = {}
    for instance in instances:
        validate_document("component-instance", instance)
        if instance["instance_id"] in by_instance:
            _fail("/instances", "duplicate instance ID", "mapping_mismatch")
        by_instance[instance["instance_id"]] = instance
    by_package = {}
    for package in packages:
        validate_document("parameter-package", package)
        if package["package_id"] in by_package:
            _fail("/packages", "duplicate package ID", "mapping_mismatch")
        by_package[package["package_id"]] = package
    assigned = {row["instance_id"]: row for row in recipe["assignments"]}
    if (len(assigned) != len(recipe["assignments"])
            or set(assigned) != {iid for iid, row in by_instance.items() if row["included"]}):
        _fail("/assignments", "included instance coverage differs", "mapping_mismatch")
    assigned_packages = {row["package_id"] for row in assigned.values()}
    if (not assigned_packages <= set(by_package)
            or not isinstance(assets_by_package, dict)
            or set(assets_by_package) != assigned_packages
            or assembly_report.get("package_sha256") !=
            {pid: canonical_sha256(by_package[pid]) for pid in assigned_packages}):
        _fail("/packages", "assembly receipt/package asset coverage differs",
              "mapping_mismatch")
    validated = {}
    for iid, row in assigned.items():
        instance = by_instance.get(iid)
        state = by_state.get(row["state_id"])
        package = by_package.get(row["package_id"])
        if (instance is None or state is None or package is None
                or instance["state_id"] != state["state_id"]
                or package["state_id"] != state["state_id"]
                or row["instance_sha256"] != canonical_sha256(instance)
                or row["state_sha256"] != canonical_sha256(state)
                or row["package_sha256"] != canonical_sha256(package)):
            _fail(f"/assignments/{iid}", "state/instance/package binding differs",
                  "mapping_mismatch")
        validated[row["package_id"]] = validate_package_assets(
            state, package, assets_by_package[row["package_id"]])
    receipts = manifest["pose_receipts_sha256"]
    if (not isinstance(receipts, dict) or not set(receipts) <= set(assigned)
            or any(type(value) is not str or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in receipts.values())):
        _fail("/manifest/pose_receipts_sha256", "invalid bound pose receipt inventory",
              "mapping_mismatch")
    rows = manifest["particles"]
    maps = {iid: {} for iid in assigned}
    seen_source = set()
    for index, particle in enumerate(rows):
        if (not isinstance(particle, dict) or set(particle) !=
                {"particle_index", "origin", "instance_id", "state_atom_id",
                 "source_atom_id", "residue_index", "name", "element"}
                or particle["particle_index"] != index
                or type(particle["residue_index"]) is not int
                or particle["residue_index"] < 0):
            _fail(f"/manifest/particles/{index}", "particle identity row is invalid",
                  "mapping_mismatch")
        origin = particle["origin"]
        source_id = particle["source_atom_id"]
        if source_id is not None:
            if type(source_id) is not str or source_id in seen_source:
                _fail(f"/manifest/particles/{index}/source_atom_id",
                      "source atom identity is repeated", "mapping_mismatch")
            seen_source.add(source_id)
        if origin == "component":
            iid = particle["instance_id"]
            sid = particle["state_atom_id"]
            if iid not in maps or type(sid) is not str or sid in maps[iid]:
                _fail(f"/manifest/particles/{index}", "component atom map is duplicated",
                      "mapping_mismatch")
            state = by_state[assigned[iid]["state_id"]]
            expected = next((atom for atom in state["atoms"] if atom["id"] == sid), None)
            binding = {row["state_atom_id"]: row["source_atom_id"]
                       for row in by_instance[iid]["atom_bindings"]}
            if (expected is None or particle["element"] != expected["element"]
                    or source_id != binding.get(sid)):
                _fail(f"/manifest/particles/{index}", "reviewed component/source atom differs",
                      "mapping_mismatch")
            maps[iid][sid] = index
        elif origin in ("retained_water", "generated_water", "generated_ion"):
            if (particle["instance_id"] is not None or particle["state_atom_id"] is not None
                    or (origin == "retained_water" and source_id is None)
                    or (origin != "retained_water" and source_id is not None)):
                _fail(f"/manifest/particles/{index}", "solvent/ion source map differs",
                      "mapping_mismatch")
        else:
            _fail(f"/manifest/particles/{index}/origin", "unsupported particle origin",
                  "capability_unavailable")
    for iid, mapping in maps.items():
        expected = {atom["id"] for atom in by_state[assigned[iid]["state_id"]]["atoms"]}
        if set(mapping) != expected:
            _fail(f"/manifest/particles/{iid}", "component atom map is incomplete",
                  "mapping_mismatch")
    expected_assignments = [
        {"instance_id": row["instance_id"], "state_id": row["state_id"],
         "package_id": row["package_id"], "package_sha256": row["package_sha256"],
         "particle_indices": sorted(maps[row["instance_id"]].values())}
        for row in recipe["assignments"]
    ]
    if assembly_report.get("assignments") != expected_assignments:
        _fail("/assembly_report/assignments", "assignment particle inventory differs",
              "mapping_mismatch")
    removed = assembly_report.get("source_removed_atom_ids")
    if (not isinstance(removed, list) or any(type(value) is not str for value in removed)
            or len(removed) != len(set(removed)) or set(removed) & seen_source):
        _fail("/assembly_report/source_removed_atom_ids", "source changes contradict particle map",
              "mapping_mismatch")
    if assembly_report.get("source_added_particle_indices") != [
            index for index, particle in enumerate(rows) if particle["source_atom_id"] is None]:
        _fail("/assembly_report/source_added_particle_indices",
              "new particle map differs from assembly receipt", "mapping_mismatch")
    return by_state, by_package, assigned, validated, maps


def _bound_reference_resources(resources, packages):
    if not isinstance(resources, dict):
        _fail("/reference_resources", "fixed reference resources are required",
              "invalid_reference")
    method_resources = {}
    template_hashes = set()
    for package in packages:
        for row in package["method"]["resource_hashes"]:
            previous = method_resources.setdefault(row["resource_id"], row["sha256"])
            if previous != row["sha256"]:
                _fail("/reference_resources", "assigned packages disagree on a resource hash",
                      "incompatible_package")
        template_hashes.update(row["content_sha256"] for row in package["assets"])
    has_gaff = "gaff/gaff-2.2.20.xml" in method_resources
    has_openff = "openff/openff_unconstrained-2.2.1.offxml" in method_resources
    has_sep = "amber/phosaa14SB.xml" in method_resources
    if sum((has_gaff, has_openff, has_sep)) > 1:
        _fail("/reference_resources", "mixed resource families lack a qualified reference",
              "incompatible_package")
    roles = (_RESOURCE_KEYS if has_gaff else _RESOURCE_OPENFF_KEYS
             if has_openff else _RESOURCE_SEP_KEYS if has_sep else _RESOURCE_PROTEIN_KEYS)
    if set(resources) != set(roles):
        _fail("/reference_resources", "reference roles differ from assigned methods",
              "incompatible_package")
    hashes = {role: _hash(_safe_forcefield_xml(resources[role], role)) for role in roles}
    roles = {"protein": "amber/ff14SB.xml" if has_sep else "amber/protein.ff14SB.xml",
             "water_ions": "amber/tip3p_standard.xml",
             "phosaa": "amber/phosaa14SB.xml",
             "gaff": "gaff/gaff-2.2.20.xml"}
    for role, resource_id in roles.items():
        if role not in hashes:
            continue
        if method_resources.get(resource_id) != hashes[role]:
            _fail(f"/reference_resources/{role}",
                  "reference XML differs from assigned package resource hash",
                  "incompatible_package")
    if has_gaff and hashes["ligand_template"] not in template_hashes:
        _fail("/reference_resources/ligand_template",
              "reference ligand template is not a package asset", "incompatible_package")
    if has_openff and hashes["ligand_template"] not in template_hashes:
        _fail("/reference_resources/ligand_template",
              "Sage template is not a package asset", "incompatible_package")
    if has_openff and method_resources[
            "openff/openff_unconstrained-2.2.1.offxml"] not in template_hashes:
        _fail("/reference_resources", "Sage OFFXML is not a package asset",
              "incompatible_package")
    return hashes


def verify_component_system_qc(assembly, recipe, states, instances, packages,
                               assets_by_package, reference_resources, *,
                               expected_manifest_sha256):
    """Reopen exact assembly bytes and compare them to independent force inputs.

    Return canonical QC report bytes only after complete term, mapping, charge,
    box, CPU energy and CPU force agreement. ``expected_manifest_sha256`` is a
    caller-trusted byte identity, required for this readback entrypoint.
    """
    if (type(expected_manifest_sha256) is not str or len(expected_manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_manifest_sha256)):
        _fail("/manifest", "a trusted canonical manifest SHA-256 is required",
              "invalid_reference")
    manifest, topology_document, assembly_report, positions, system_xml = (
        _bound_assembly_bytes(assembly, expected_manifest_sha256))
    by_state, by_package, assigned, validated, maps = _bind_scientific_inputs(
        manifest, assembly_report, recipe, states, instances, packages, assets_by_package)
    topology = topology_from_document(topology_document, manifest)
    subject = _deserialize_bounded_system(system_xml)
    inventory = snapshot_system_terms(subject)
    if inventory["particle_count"] != len(manifest["particles"]):
        _fail("/system/particles", "System particle count differs from manifest",
              "mapping_mismatch")
    box_summary = {"max_term_abs_delta": 0.0}
    _compare(inventory["box_vectors_nm"], manifest["box_vectors_nm"],
             "/system/box_vectors_nm", box_summary, box=True)
    report_system = assembly_report.get("system")
    if (not isinstance(report_system, dict)
            or report_system.get("particle_count") != inventory["particle_count"]
            or sorted(report_system.get("force_types", [])) != inventory["force_types"]
            or report_system.get("constraint_count") != len(inventory["constraints"])
            or report_system.get("exception_count") != len(inventory["exceptions"])
            or report_system.get("virtual_site_count") != 0):
        _fail("/assembly_report/system", "claimed System inventory differs from reopened bytes",
              "mapping_mismatch")
    actual_charge = sum(row["charge_e"] for row in inventory["particles"])
    claim_charge = assembly_report.get("net_charge_e")
    if (type(claim_charge) not in (int, float) or not math.isfinite(claim_charge)
            or abs(actual_charge - claim_charge) > 1e-6
            or (recipe["assembly"]["neutralize"] and abs(actual_charge) > 1e-6)):
        _fail("/assembly_report/net_charge_e", "net charge differs from System/recipe")
    solvent = assembly_report.get("solvent")
    rows = manifest["particles"]
    expected_solvent = {
        "retained_water": len({row["residue_index"] for row in rows
                               if row["origin"] == "retained_water"}),
        "generated_water": len({row["residue_index"] for row in rows
                                if row["origin"] == "generated_water"}),
        "sodium": sum(row["origin"] == "generated_ion" and row["element"] == "Na"
                      for row in rows),
        "chloride": sum(row["origin"] == "generated_ion" and row["element"] == "Cl"
                        for row in rows),
    }
    if (not isinstance(solvent, dict)
            or any(solvent.get(key) != value for key, value in expected_solvent.items())
            or any(row["origin"] == "generated_ion" and row["element"] not in ("Na", "Cl")
                   for row in rows)):
        _fail("/assembly_report/solvent", "solvent/ion inventory differs from particle map",
              "mapping_mismatch")
    assigned_package_docs = [by_package[pid] for pid in sorted(
        {row["package_id"] for row in assigned.values()})]
    resource_hashes = _bound_reference_resources(reference_resources, assigned_package_docs)
    assembly_resources = assembly_report.get("resources")
    expected_resources = ([
        ("amber/ff14SB.xml", resource_hashes["protein"]),
        ("amber/phosaa14SB.xml", resource_hashes["phosaa"]),
        ("amber/tip3p_standard.xml", resource_hashes["water_ions"]),
    ] if "phosaa" in resource_hashes else [
        ("amber/protein.ff14SB.xml", resource_hashes["protein"]),
        ("amber/tip3p_standard.xml", resource_hashes["water_ions"]),
    ])
    if (not isinstance(assembly_resources, list)
            or len(assembly_resources) != len(expected_resources)
            or [(row.get("resource_id"), row.get("sha256"))
                for row in assembly_resources if isinstance(row, dict)] != expected_resources):
        _fail("/assembly_report/resources", "assembly resource digests differ from reference",
              "incompatible_package")
    reference, built_hashes = build_direct_forcefield_reference(topology, reference_resources)
    if built_hashes != resource_hashes:
        _fail("/reference_resources", "reference resource bytes changed during construction",
              "invalid_reference")
    component_checks = {}
    max_package_delta = 0.0
    for iid, assignment in assigned.items():
        pid = assignment["package_id"]
        check = compare_package_force_subset(
            subject, validated[pid]["parameters"], maps[iid],
            net_formal_charge=by_state[assignment["state_id"]]["net_formal_charge"])
        check["package_id"] = pid
        component_checks[iid] = check
        max_package_delta = max(max_package_delta, check["max_term_abs_delta"])
    term_check = compare_system_terms(subject, reference)
    observables = compare_cpu_observables(subject, reference, positions)
    checks = ["exact assembly artifact hashes and canonical records reopened",
              "state, instance, package and source particle maps agree",
              "source package force terms and direct ForceField reference agree",
              "periodic box, solvent/ion inventory and net charge agree",
              "CPU potential energy and every force component agree"]
    qc = {
        "format": "neomd-component-system-qc-v1",
        "status": "numerically_executable",
        "assembly_manifest_sha256": expected_manifest_sha256,
        "recipe_sha256": canonical_sha256(recipe),
        "source_content_sha256": recipe["source_topology"]["content_sha256"],
        "package_sha256": {pid: canonical_sha256(by_package[pid])
                           for pid in sorted({row["package_id"] for row in assigned.values()})},
        "reference": {"producer": "direct-openmm-forcefield-v1", "platform": "CPU",
                      "threads": 1, "resource_order": list(resource_hashes),
                      "resource_sha256": resource_hashes},
        "tolerances": {"term_abs": _TERM_ABS, "term_rel": _TERM_REL,
                       "box_abs_nm": _BOX_ABS, "energy_abs_kj_mol": _ENERGY_ABS,
                       "energy_rel": _ENERGY_REL,
                       "force_abs_kj_mol_nm": _FORCE_ABS, "force_rel": _FORCE_REL,
                       "charge_abs_e": 1e-6},
        "inventory": {"particle_count": inventory["particle_count"],
                      "force_types": inventory["force_types"],
                      "constraint_count": len(inventory["constraints"]),
                      "exception_count": len(inventory["exceptions"]),
                      "virtual_site_count": 0,
                      "retained_water": expected_solvent["retained_water"],
                      "generated_water": expected_solvent["generated_water"],
                      "sodium": expected_solvent["sodium"],
                      "chloride": expected_solvent["chloride"]},
        "net_charge_e": actual_charge,
        "component_checks": component_checks,
        "comparison": {"max_system_term_abs_delta": term_check["max_term_abs_delta"],
                       "max_package_term_abs_delta": max_package_delta,
                       **observables},
        "checks": checks,
    }
    encoded = rfc8785.dumps(qc)
    read_component_system_qc_report(encoded, expected_sha256=_hash(encoded))
    return encoded


def read_component_system_qc_report(report_bytes, *, expected_sha256):
    """Validate a retained v1 QC receipt and its exact encoded byte identity."""
    if (type(report_bytes) is not bytes or type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
            or _hash(report_bytes) != expected_sha256):
        _fail("/qc_report", "QC report byte identity differs", "invalid_reference")
    report = _canonical_json(report_bytes, "/qc_report")
    error = next(iter(_QC_VALIDATOR.iter_errors(report)), None)
    if error is not None:
        pointer = "".join("/" + str(part).replace("~", "~0").replace("/", "~1")
                          for part in error.absolute_path)
        _fail("/qc_report" + pointer, "QC receipt fails versioned schema",
              "contract_version_mismatch")
    reference = report["reference"]
    if set(reference["resource_order"]) != set(reference["resource_sha256"]):
        _fail("/qc_report/reference/resource_sha256",
              "reference resource hashes differ from ordered roles",
              "contract_version_mismatch")
    if any(row["package_id"] not in report["package_sha256"]
           for row in report["component_checks"].values()):
        _fail("/qc_report/component_checks", "component refers to an unbound package",
              "mapping_mismatch")
    comparison = report["comparison"]
    if (comparison["max_package_term_abs_delta"] !=
            max(row["max_term_abs_delta"] for row in report["component_checks"].values())
            or not math.isclose(comparison["energy_abs_delta_kj_mol"],
                                abs(comparison["subject_energy_kj_mol"] -
                                    comparison["reference_energy_kj_mol"]),
                                rel_tol=0, abs_tol=1e-9)):
        _fail("/qc_report/comparison", "reported numeric deltas contradict observations",
              "invalid_reference")
    return report
