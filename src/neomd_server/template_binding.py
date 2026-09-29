"""Read-only source-lineage binding for versioned scientific templates."""

from __future__ import annotations

import copy
import io
import json
import sysconfig
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid5

import jsonschema
import numpy as np

from neomd.prepared import file_digest, read_prepared_bundle

from .contract import ContractError, canonical_bytes, content_hash
from .protocols import _GROUPS, ScientificLimits, compile_protocol, particle_identity_hash

__all__ = ["bind_protocol_template", "write_component_topology_bridge"]


@lru_cache(maxsize=1)
def _validator():
    root = Path(__file__).resolve().parents[2] / "contracts/scientific-protocol-template/v1"
    if not root.exists():
        root = (
            Path(sysconfig.get_path("data"))
            / "share/neodynamics/contracts/scientific-protocol-template/v1"
        )
    schema = json.loads((root / "bind.schema.json").read_text())
    return jsonschema.Draft202012Validator(
        {"$ref": "#/$defs/TemplateBindRequest", "$defs": schema["$defs"]},
        format_checker=jsonschema.FormatChecker(),
    )


def _validate(request):
    problems = [
        (
            "/" + "/".join(map(str, error.absolute_path)),
            f"does not satisfy {error.validator or 'closed schema'}",
        )
        for error in _validator().iter_errors(request)
    ]
    try:
        canonical_bytes(request)
    except (ValueError, TypeError):
        problems.append(("/", "must be finite interoperable canonical JSON"))
    if problems:
        raise ContractError(problems)


def _diagnostic(code, path, message):
    return {"code": code, "path": path, "message": message}


def _artifact(root, manifest, name):
    entry = manifest["artifacts"].get(name)
    if entry is None:
        raise ValueError(f"required prepared artifact {name} is absent")
    target = (root / name).resolve(strict=True)
    if not target.is_relative_to(root) or target.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"prepared artifact {name} differs from its manifest")
    if file_digest(target) != entry["sha256"]:
        raise ValueError(f"prepared artifact {name} differs from its manifest")
    return target


def write_component_topology_bridge(manifest, prepared_directory, target):
    """Write a checked v2 PDBx execution carrier without modifying the bundle.

    The JSON topology and exact particle coordinates remain authoritative. This
    bridge is used only for the existing Plan's PDBx input-file interface.
    """
    from openmm import app, unit

    from neomd.component_system_qc import _topology_from_document

    root = Path(prepared_directory).resolve(strict=True)
    if manifest.get("schema_version") != 2 or "topology.pdbx" in manifest["artifacts"]:
        raise ValueError("topology bridge requires a component-prepared/v2 bundle")
    assembly = json.loads(_artifact(root, manifest, "assembly/manifest.json").read_text())
    document = json.loads(_artifact(root, manifest, "assembly/topology.json").read_text())
    atom_rows = manifest["atoms"]
    if (
        len(atom_rows) != manifest["particle_count"]
        or len(document["atoms"]) != len(atom_rows)
        or manifest["topology_particle_indices"] != list(range(len(atom_rows)))
    ):
        raise ValueError("component topology and prepared particle order differ")
    for index, (atom, row, particle) in enumerate(
        zip(atom_rows, document["atoms"], manifest["particles"])
    ):
        if (
            row["particle_index"] != index
            or atom["atom_id"] != particle["identity_id"]
            or atom["name"] != row["name"]
            or atom["element"] != row["element"]
        ):
            raise ValueError("component topology and prepared atom identity differ")
    atom_indices = {row["atom_id"]: index for index, row in enumerate(atom_rows)}
    graph = {
        tuple(sorted((atom_indices[a], atom_indices[b])))
        for a, b in (bond["atom_ids"] for bond in manifest["bonds"])
    }
    if graph != {tuple(pair) for pair in document["bonds"]}:
        raise ValueError("component topology and prepared graph differ")
    topology = _topology_from_document(document, assembly)
    positions = np.fromfile(
        _artifact(root, manifest, manifest["coordinates"]["artifact"]), dtype="<f8"
    ).reshape((-1, 3))
    if len(positions) != len(atom_rows) or not np.isfinite(positions).all():
        raise ValueError("component coordinates differ from prepared particle order")
    stream = io.StringIO()
    app.PDBxFile.writeFile(topology, positions * unit.nanometer, stream, keepIds=True)
    candidate = stream.getvalue()
    decoded = app.PDBxFile(io.StringIO(candidate))
    actual_atoms = list(decoded.topology.atoms())
    if (
        len(actual_atoms) != len(atom_rows)
        or [(atom.name, atom.element.symbol) for atom in actual_atoms]
        != [(row["name"], row["element"]) for row in atom_rows]
        or {tuple(sorted((a.index, b.index))) for a, b in decoded.topology.bonds()} != graph
    ):
        raise ValueError("PDBx bridge does not preserve the prepared topology")
    decoded_positions = np.asarray(decoded.positions.value_in_unit(unit.nanometer))
    if not np.allclose(decoded_positions, positions, rtol=0, atol=0.0000051):
        raise ValueError(
            "PDBx bridge does not preserve prepared coordinates within writer precision"
        )
    declared_box = manifest["box_vectors_nm"]
    decoded_box = decoded.topology.getPeriodicBoxVectors()
    if (declared_box is None) != (decoded_box is None):
        raise ValueError("PDBx bridge periodicity differs from the prepared System")
    if declared_box is not None:
        actual_box = np.asarray(
            [list(vector.value_in_unit(unit.nanometer)) for vector in decoded_box]
        )
        if not np.allclose(actual_box, declared_box, rtol=0, atol=0.0000051):
            raise ValueError("PDBx bridge box differs from the prepared System")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = candidate.encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise ValueError("immutable topology bridge has different bytes")
    else:
        target.write_bytes(encoded)
    return target


def _select_atom(selector, index, manifest, mapping, report, assembly):
    path = f"/template/atom_selectors/{index}"
    sources = [
        row
        for row in manifest["sources"]
        if row["content_sha256"] == selector["source_content_sha256"]
    ]
    if not sources:
        return None, _diagnostic(
            "source_mismatch",
            path + "/source_content_sha256",
            "source revision is absent from this prepared System",
        )
    source_rows = [
        (source, atom)
        for source in sources
        for atom in source["atoms"]
        if atom["source_atom_id"] == selector["source_atom_id"]
        and atom["disposition"] == "retained"
    ]
    if not source_rows:
        return None, _diagnostic(
            "missing_atom",
            path + "/source_atom_id",
            "source atom is absent or removed in this prepared System",
        )
    atoms = {row["atom_id"]: row for row in manifest["atoms"]}
    mapped = {row["atom_id"]: row for row in mapping["atoms"]}
    particles = {row["identity_id"]: row for row in manifest["particles"]}
    assignments = [
        row for row in report["assignments"] if row["instance_id"] == selector["instance_id"]
    ]
    if not assignments:
        return None, _diagnostic(
            "instance_mismatch",
            path + "/instance_id",
            "component instance is absent from this prepared System",
        )
    if len(assignments) != 1:
        return None, _diagnostic(
            "ambiguous_atom", path + "/instance_id", "component instance has ambiguous assignments"
        )
    assignment = assignments[0]
    if assignment["state_id"] != selector["state_id"]:
        return None, _diagnostic(
            "state_mismatch",
            path + "/state_id",
            "component reviewed state changed, including possible protonation",
        )
    scope = set(assignment["particle_indices"])
    candidates = []
    for source, source_atom in source_rows:
        atom_id = source_atom["prepared_atom_id"]
        atom, row, particle = atoms.get(atom_id), mapped.get(atom_id), particles.get(atom_id)
        if (
            atom is None
            or row is None
            or particle is None
            or atom["source_id"] != source["source_id"]
            or atom["source_atom_id"] != selector["source_atom_id"]
            or row["source_id"] != source["source_id"]
            or row["source_atom_id"] != selector["source_atom_id"]
        ):
            continue
        position = particle["particle_index"]
        if (
            position in scope
            and position < len(assembly["particles"])
            and assembly["particles"][position]["instance_id"] == selector["instance_id"]
            and str(
                uuid5(
                    UUID(source["source_id"]),
                    f"atom:{assembly['particles'][position]['source_atom_id']}",
                )
            )
            == selector["source_atom_id"]
        ):
            candidates.append((atom, particle))
    if not candidates:
        return None, _diagnostic(
            "missing_atom",
            path + "/source_atom_id",
            "source atom has no exact prepared mapping in the selected instance",
        )
    if len(candidates) != 1:
        return None, _diagnostic(
            "ambiguous_atom",
            path + "/source_atom_id",
            "source atom maps to multiple prepared atoms in the selected instance",
        )
    atom, particle = candidates[0]
    if (
        atom["element"] != selector["expected_element"]
        or particle["identity_kind"] != "atom"
        or not particle["control_eligible"]
        or particle["mass_dalton"] <= 0
    ):
        return None, _diagnostic(
            "ineligible_atom",
            path,
            "selected atom changed element or is not a massive control-eligible atom",
        )
    return {
        "selector_id": selector["selector_id"],
        "source_content_sha256": selector["source_content_sha256"],
        "instance_id": selector["instance_id"],
        "state_id": selector["state_id"],
        "source_atom_id": selector["source_atom_id"],
        "prepared_atom_id": atom["atom_id"],
        "particle_index": particle["particle_index"],
    }, None


def bind_protocol_template(
    request, *, prepared_directory, limits: ScientificLimits, reference_resolver=None
):
    """Resolve a template against exact retained lineage; return no durable state."""
    _validate(request)
    prepared, template = request["prepared"], request["template"]
    template_sha = content_hash(template)

    def blocked(findings):
        return {
            "schema_version": 1,
            "status": "blocked",
            "prepared": prepared,
            "template_sha256": template_sha,
            "diagnostics": findings,
        }

    root = Path(prepared_directory).resolve(strict=True)
    try:
        manifest = read_prepared_bundle(
            root / "manifest.json", expected_sha256=prepared["manifest_sha256"]
        )
        mapping_path = _artifact(root, manifest, "mapping.json")
        if file_digest(mapping_path) != prepared["mapping_sha256"]:
            raise ValueError("prepared mapping digest differs from binding")
        mapping = json.loads(mapping_path.read_text())
        report = json.loads(_artifact(root, manifest, "assembly/report.json").read_text())
        assembly = json.loads(_artifact(root, manifest, "assembly/manifest.json").read_text())
        if (
            manifest["prepared_revision_id"] != prepared["prepared_revision_id"]
            or mapping["prepared_revision_id"] != prepared["prepared_revision_id"]
            or mapping["atoms"]
            != [
                {key: atom[key] for key in ("atom_id", "residue_id", "source_id", "source_atom_id")}
                for atom in manifest["atoms"]
            ]
            or report["source_content_sha256"] != assembly["source_content_sha256"]
            or len(assembly["particles"]) != manifest["particle_count"]
        ):
            raise ValueError("prepared lineage and assembly report disagree")
    except (ValueError, OSError, KeyError, TypeError, IndexError):
        return blocked(
            [
                _diagnostic(
                    "prepared_incompatible",
                    "/template",
                    "prepared bundle, mapping or assembly lineage is unavailable or changed",
                )
            ]
        )
    profile = report.get("assembly", {})
    if (
        manifest["schema_version"] != 2
        or manifest["parameters"].get("component_qc_status") != "numerically_executable"
        or profile.get("constraints") != "HBonds"
        or profile.get("hydrogen_mass_amu") != 4
        or template["protocol"]["physics"]["timestep_ps"] > 0.002
    ):
        return blocked(
            [
                _diagnostic(
                    "prepared_incompatible",
                    "/template/compatibility",
                    "requires QC-passed HBonds, 4 amu HMR and a timestep at most 0.002 ps",
                )
            ]
        )
    if "barostat" in template["protocol"]["physics"] and manifest["box_vectors_nm"] is None:
        return blocked(
            [
                _diagnostic(
                    "prepared_incompatible",
                    "/template/protocol/physics/barostat",
                    "pressure requires a periodic prepared System",
                )
            ]
        )
    selectors = {row["selector_id"]: row for row in template["atom_selectors"]}
    if len(selectors) != len(template["atom_selectors"]):
        return blocked(
            [
                _diagnostic(
                    "protocol_invalid",
                    "/template/atom_selectors",
                    "logical selector IDs must be unique",
                )
            ]
        )
    resolved, diagnostics = {}, []
    for index, selector in enumerate(template["atom_selectors"]):
        selected, finding = _select_atom(selector, index, manifest, mapping, report, assembly)
        if finding:
            diagnostics.append(finding)
        else:
            resolved[selector["selector_id"]] = selected
    if diagnostics:
        return blocked(diagnostics)
    descriptor = copy.deepcopy(template["protocol"])
    descriptor["prepared_revision_id"] = prepared["prepared_revision_id"]
    if "replica_seed" in request:
        descriptor["physics"]["seed"] = request["replica_seed"]
    ordered = []

    def convert(parameters, path):
        for key, value in parameters.items():
            if key in _GROUPS:
                for index, selector_id in enumerate(value):
                    if selector_id not in resolved:
                        diagnostics.append(
                            _diagnostic(
                                "missing_atom",
                                f"{path}/{key}/{index}",
                                "logical selector has no source anchor",
                            )
                        )
                    else:
                        parameters[key][index] = resolved[selector_id]["prepared_atom_id"]
                        if selector_id not in ordered:
                            ordered.append(selector_id)
            elif key == "params":
                for index, pair in enumerate(value):
                    convert(pair, f"{path}/params/{index}")

    for section in ("colvars", "restraints", "steering"):
        for index, definition in enumerate(descriptor.get(section, [])):
            convert(definition["parameters"], f"/template/protocol/{section}/{index}/parameters")
    if set(ordered) != set(selectors):
        diagnostics.append(
            _diagnostic(
                "protocol_invalid",
                "/template/atom_selectors",
                "every source selector must be used by an atom group",
            )
        )
    reference_map = {
        row["template_reference_id"]: row["reference_id"] for row in request["reference_bindings"]
    }
    if len(reference_map) != len(request["reference_bindings"]):
        diagnostics.append(
            _diagnostic(
                "reference_incompatible",
                "/reference_bindings",
                "template reference keys must be unique",
            )
        )
    used_references, reference_rows = set(), []
    particle_sha = particle_identity_hash(manifest)
    for section in ("colvars", "restraints", "steering"):
        for index, definition in enumerate(descriptor.get(section, [])):
            reference = definition["parameters"].get("reference")
            if reference is None:
                continue
            path = f"/template/protocol/{section}/{index}/parameters/reference"
            result_path = path.removeprefix("/template")
            if reference["kind"] == "prepared_start":
                reference_rows.append(
                    {
                        "path": result_path,
                        "kind": "prepared_start",
                        "reference_id": None,
                        "content_sha256": manifest["artifacts"][
                            manifest["coordinates"]["artifact"]
                        ]["sha256"],
                        "particle_identity_sha256": particle_sha,
                    }
                )
            else:
                key = reference["reference_id"]
                if key not in reference_map:
                    diagnostics.append(
                        _diagnostic(
                            "reference_incompatible",
                            path,
                            "target-owned registered reference binding is missing",
                        )
                    )
                    continue
                used_references.add(key)
                target_id = reference_map[key]
                reference["reference_id"] = target_id
                try:
                    result = reference_resolver(target_id)
                    meta = result.metadata
                    reference_rows.append(
                        {
                            "path": result_path,
                            "kind": "registered",
                            "reference_id": target_id,
                            "content_sha256": meta["content_sha256"],
                            "particle_identity_sha256": meta["particle_identity_sha256"],
                        }
                    )
                except (TypeError, ValueError, KeyError, AttributeError):
                    diagnostics.append(
                        _diagnostic(
                            "reference_incompatible",
                            path,
                            "target registered reference is unavailable",
                        )
                    )
    if set(reference_map) != used_references:
        diagnostics.append(
            _diagnostic(
                "reference_incompatible",
                "/reference_bindings",
                "unused target reference binding is not allowed",
            )
        )
    if diagnostics:
        return blocked(diagnostics)
    try:
        with tempfile.TemporaryDirectory(prefix="neomd-template-bind-") as scratch:
            scratch = Path(scratch)
            topology = write_component_topology_bridge(manifest, root, scratch / "topology.pdbx")
            compile_protocol(
                descriptor,
                prepared_manifest=manifest,
                prepared_directory=root,
                output_directory=scratch / "output",
                protocol_directory=scratch / "refs",
                reference_resolver=reference_resolver,
                limits=limits,
                materialize_references=False,
                trusted_topology_file=topology,
            )
    except (ValueError, OSError, KeyError, ContractError) as error:
        if isinstance(error, ContractError):
            findings = [
                _diagnostic("protocol_invalid", "/template/protocol" + path, message)
                for path, message in error.problems
            ]
        else:
            findings = [
                _diagnostic(
                    "prepared_incompatible",
                    "/template",
                    "prepared topology bridge or concrete protocol is incompatible",
                )
            ]
        return blocked(findings)
    return {
        "schema_version": 1,
        "status": "ready",
        "prepared": prepared,
        "template_sha256": template_sha,
        "protocol": descriptor,
        "protocol_sha256": content_hash(descriptor),
        "registration_sha256": content_hash({"prepared": prepared, "protocol": descriptor}),
        "particle_identity_sha256": particle_sha,
        "selections": [resolved[selector_id] for selector_id in ordered],
        "references": reference_rows,
        "diagnostics": [],
    }
