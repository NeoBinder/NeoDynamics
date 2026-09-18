"""Explicit protein System assembly and independent artifact-bound reports.

See docs/protein-preparation.md and contracts/assembly-report/v1.
"""

from __future__ import annotations

import copy
import io
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from uuid import UUID

import numpy as np

from .prepared import _coordinates, _error, _json_bytes, _raise, read_prepared_bundle

REPORT_ARTIFACT = "assembly-report.json"
_ASSEMBLY_ARTIFACTS = {"system.xml", "topology.pdbx", "mapping.json", "coordinates.f64le",
                       "forcefield/protein.ff14SB.xml", "forcefield/tip3p_standard.xml"}


def _report_bindings(manifest):
    """Validate manifest-derived evidence instead of trusting a copied claim."""
    revision = manifest["prepared_revision_id"]
    try:
        valid_revision = (isinstance(revision, str) and str(UUID(revision)) == revision
                          and UUID(revision).int != 0)
    except (ValueError, TypeError, AttributeError):
        valid_revision = False
    if not valid_revision:
        _raise([_error("assembly_report.prepared_revision_id", "requires a nonnil canonical UUID")])
    artifacts = manifest["artifacts"]
    if (not isinstance(artifacts, dict) or not _ASSEMBLY_ARTIFACTS <= artifacts.keys()
            or REPORT_ARTIFACT in artifacts or "manifest.json" in artifacts):
        _raise([_error("assembly_report.bindings", "requires fixed assembly artifacts without self/manifest")])
    for name, descriptor in artifacts.items():
        if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or name == "." or str(PurePosixPath(name)) != name):
            _raise([_error("assembly_report.bindings", "requires canonical relative artifact paths")])
        if (not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "size_bytes"}
                or type(descriptor["size_bytes"]) is not int or descriptor["size_bytes"] < 0
                or not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in descriptor["sha256"])):
            _raise([_error("assembly_report.bindings", "requires exact SHA256/integer size descriptors")])
    return copy.deepcopy(artifacts)


def _bind_report(evidence, manifest):
    """Finalize report bindings without importing OpenMM or hashing itself."""
    if not isinstance(evidence, dict) or set(evidence) != {"engine", "parameters", "resources", "effective"}:
        _raise([_error("assembly_report", "requires only engine, parameters, resources and effective evidence")])
    return {"schema_version": 1, "report_type": "protein_system_assembly",
            "scope": "system_only", "prepared_revision_id": manifest["prepared_revision_id"],
            "bindings": _report_bindings(manifest), **copy.deepcopy(evidence)}


def _graph(topology):
    # No label normalization: reject any writer/decoder loss before publication.
    return {"chains": [c.id for c in topology.chains()],
            "residues": [(r.chain.index, r.name, r.id, r.insertionCode)
                         for r in topology.residues()],
            "atoms": [(a.residue.index, a.name,
                       None if a.element is None else a.element.symbol, a.formalCharge)
                      for a in topology.atoms()],
            "bonds": sorted((min(a.index, b.index), max(a.index, b.index),
                             bond.order, None if bond.type is None else str(bond.type))
                            for bond in topology.bonds() for a, b in [bond])}


def _evidence(parameters, resources, effective):
    from .prepare import _openmm

    openmm, _, _, _ = _openmm()
    return {"engine": {"name": "openmm", "version": openmm.version.full_version},
            "parameters": parameters, "resources": [row for row, _ in resources],
            "effective": effective}


def assemble_protein_system(topology, positions_nm, directory, *, lineage, parameters):
    """Build/export a System from an already chemically prepared v2 topology.

    All eleven System settings are required explicitly. This adds no atoms,
    solvent, ions, protonation or simulation controls. Caller topology, coordinates
    and lineage remain unchanged. The report certifies replayable System assembly,
    not preparation history, sequence completeness or simulation readiness.
    The PDBx graph must round-trip losslessly; exact positions/box remain in the
    bundle's f64/manifest artifacts. Returns an immutable local SystemBundle only
    after report and actual System verification; no remote durability guarantee.
    """
    from .prepare import _openmm, _particle_inventory
    from .system import SystemBundle
    from .system_assembly import _assemble

    if getattr(lineage, "schema_version", None) != 2:
        _raise([_error("lineage", "System assembly requires prepared lineage v2")])
    copied_lineage, candidate = copy.deepcopy((lineage, topology))
    coords = _coordinates(positions_nm).copy()
    copied_lineage.bind_assembled(candidate, coords)
    _, app, unit, serializer = _openmm()
    stream = io.StringIO()
    app.PDBxFile.writeFile(candidate, coords * unit.nanometer, stream, keepIds=True)
    topology_bytes = stream.getvalue().encode("utf-8")
    decoded = app.PDBxFile(io.StringIO(stream.getvalue()))
    if _graph(candidate) != _graph(decoded.topology):
        _raise([_error("topology", "PDBx export loses topology graph or labels")])
    decoded.topology.setPeriodicBoxVectors(candidate.getPeriodicBoxVectors())
    system, requested, resources, effective = _assemble(decoded.topology, parameters)
    inventory, particle_coords = _particle_inventory(system, coords * unit.nanometer)
    evidence = _evidence(requested, resources, effective)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    final = root / copied_lineage.revision_id
    if final.exists():
        _raise([_error("prepared_revision_id", "immutable revision already exists")])
    with tempfile.TemporaryDirectory(prefix=".assembly-", dir=root) as temporary:
        work = Path(temporary)
        files = {"topology.pdbx": topology_bytes,
                 "system.xml": serializer.serialize(system).encode("utf-8"),
                 **{metadata["artifact"]: data for metadata, data in resources}}
        artifacts = {}
        for name, data in files.items():
            target = work / "inputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            artifacts[name] = target
        manifest_path = copied_lineage.publish(
            work / "prepared", topology=candidate, positions_nm=coords,
            particles=inventory, coordinates_nm=particle_coords,
            box_vectors_nm=effective["box_vectors_nm"], artifacts=artifacts,
            parameters={"system_scope": "prepared System before runtime modifications",
                        "system_constraints": system.getNumConstraints(),
                        "force_types": [type(force).__name__ for force in system.getForces()]},
            assembly_report=evidence,
        )
        verify_protein_system_assembly(manifest_path)
        os.rename(Path(manifest_path).parent, final)
    return SystemBundle(topology_file=str(final / "topology.pdbx"),
                        system_xml=str(final / "system.xml"),
                        prepared_manifest=str(final / "manifest.json"))


def _read_report(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate report key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite report number: {value}")

    try:
        return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid assembly report JSON") from error


def verify_protein_system_assembly(manifest_path, *, expected_sha256=None):
    """Verify report bindings and rebuild the exact System before deserializing it.

    Supply a caller-trusted manifest SHA256 for untrusted artifacts. No hash
    authenticates an otherwise untrusted producer or preparation history. Only
    the qualified installed OpenMM and exact approved ff14SB/TIP3P resources
    are replayed; captured XML never authorizes code or resource loading.
    Run in host-owned bounded isolation for untrusted intake. This SDK function
    does not enable broader v2 service intake. Returns the verified report.
    """
    from .prepare import _openmm, verify_prepared_system
    from .system_assembly import _assemble, _resource_snapshot

    manifest = read_prepared_bundle(manifest_path, expected_sha256=expected_sha256)
    if (manifest["schema_version"] != 2
            or manifest["parameters"].get("assembly_report_artifact") != REPORT_ARTIFACT
            or REPORT_ARTIFACT not in manifest["artifacts"]):
        _raise([_error("assembly_report", "requires v2 with a bound assembly report")])
    root = Path(manifest_path).resolve().parent
    report = _read_report(root / REPORT_ARTIFACT)
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int:
        _raise([_error("assembly_report", "invalid report object/version")])
    # Reconstruct the closed report shape rather than trusting claimed effects.
    resources = _resource_snapshot()
    for metadata, data in resources:
        name = metadata["artifact"]
        if name not in manifest["artifacts"] or (root / name).read_bytes() != data:
            _raise([_error("assembly_report.resources", "resource differs from approved installed bytes")])
    if ("topology.pdbx" not in manifest["artifacts"]
            or "topology.pdb" in manifest["artifacts"]
            or "system.xml" not in manifest["artifacts"]):
        _raise([_error("assembly_report", "requires exact PDBx topology and System artifacts")])
    _, app, unit, serializer = _openmm()
    topology = app.PDBxFile(str(root / "topology.pdbx")).topology
    box = manifest["box_vectors_nm"]
    if box is None:
        _raise([_error("box_vectors_nm", "PME assembly requires an explicit periodic box")])
    # PDBx cell lengths/angles are decimal-quantized, not the authoritative box.
    topology.setPeriodicBoxVectors(np.asarray(box) * unit.nanometer)
    system, requested, replay_resources, effective = _assemble(topology, report.get("parameters"))
    if (root / "system.xml").read_bytes() != serializer.serialize(system).encode("utf-8"):
        _raise([_error("assembly_report.system", "System differs from exact forcefield replay")])
    binding_manifest = {**manifest, "artifacts": {name: descriptor for name, descriptor
                        in manifest["artifacts"].items() if name != REPORT_ARTIFACT}}
    expected = _bind_report(_evidence(requested, replay_resources, effective), binding_manifest)
    if _json_bytes(report) != _json_bytes(expected):
        _raise([_error("assembly_report", "report shape, bindings or actual assembly evidence differs")])
    verify_prepared_system(manifest_path, expected_sha256=expected_sha256)
    return report
