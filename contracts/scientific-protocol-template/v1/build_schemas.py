"""Generate closed template/bind schemas from the registered protocol vocabulary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACTS = HERE.parents[1]
UUID = {"type": "string", "format": "uuid",
        "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}


def _object(properties, required):
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}


def _write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    concrete = json.loads((CONTRACTS / "scientific-protocol/v1/protocol.schema.json").read_text())
    prepared = json.loads((CONTRACTS / "imd/v1/protocol.schema.json").read_text())["$defs"]["PreparedBinding"]
    template_protocol = copy.deepcopy(concrete)
    for key in ("$schema", "$id", "title", "description"):
        template_protocol.pop(key, None)
    template_protocol["properties"].pop("prepared_revision_id")
    template_protocol["required"].remove("prepared_revision_id")
    selector = _object({"selector_id": UUID, "source_content_sha256": SHA,
                        "instance_id": UUID, "state_id": UUID, "source_atom_id": UUID,
                        "expected_element": {"type": "string", "pattern": "^[A-Z][a-z]?$"}},
                       ["selector_id", "source_content_sha256", "instance_id", "state_id",
                        "source_atom_id", "expected_element"])
    template = _object({
        "schema_version": {"const": 1},
        "protocol": template_protocol,
        "atom_selectors": {"type": "array", "items": selector, "maxItems": 1024},
        "compatibility": _object({"profile": {"const": "hbonds-hmr4-2fs-v1"}}, ["profile"]),
    }, ["schema_version", "protocol", "atom_selectors", "compatibility"])
    template.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                     "$id": "urn:neodyn:scientific-protocol-template:v1",
                     "title": "NeoDynamics scientific protocol template v1"})
    _write("template.schema.json", template)

    template_ref = copy.deepcopy(template)
    for key in ("$schema", "$id", "title"):
        template_ref.pop(key, None)
    concrete_ref = copy.deepcopy(concrete)
    for key in ("$schema", "$id", "title", "description"):
        concrete_ref.pop(key, None)
    reference_binding = _object({"template_reference_id": UUID, "reference_id": UUID},
                                ["template_reference_id", "reference_id"])
    request = _object({
        "prepared": prepared, "template": template_ref,
        "reference_bindings": {"type": "array", "items": reference_binding, "maxItems": 32},
        "replica_seed": {"type": "integer", "minimum": 0, "maximum": 2147483647,
                         "description": "Override only the concrete protocol seed; template SHA-256 remains the unmodified base template."},
    }, ["prepared", "template", "reference_bindings"])
    diagnostic = _object({"code": {"enum": ["missing_atom", "ambiguous_atom", "ineligible_atom",
                                                  "source_mismatch", "instance_mismatch", "state_mismatch",
                                                  "prepared_incompatible", "reference_incompatible",
                                                  "protocol_invalid"]},
                          "path": {"type": "string", "pattern": "^/(template|reference_bindings)(/.*)?$"},
                          "message": {"type": "string", "minLength": 1, "maxLength": 512}},
                         ["code", "path", "message"])
    selection = _object({"selector_id": UUID, "source_content_sha256": SHA,
                         "instance_id": UUID, "state_id": UUID, "source_atom_id": UUID,
                         "prepared_atom_id": UUID, "particle_index": {"type": "integer", "minimum": 0}},
                        ["selector_id", "source_content_sha256", "instance_id", "state_id",
                         "source_atom_id", "prepared_atom_id", "particle_index"])
    reference = _object({"path": {"type": "string", "pattern": "^/protocol/.*$"},
                         "kind": {"enum": ["prepared_start", "registered"]},
                         "reference_id": {"oneOf": [UUID, {"type": "null"}]},
                         "content_sha256": SHA, "particle_identity_sha256": SHA},
                        ["path", "kind", "reference_id", "content_sha256",
                         "particle_identity_sha256"])
    result = _object({
        "schema_version": {"const": 1}, "status": {"enum": ["ready", "blocked"]},
        "prepared": prepared, "template_sha256": SHA,
        "diagnostics": {"type": "array", "items": diagnostic, "maxItems": 256},
        "protocol": concrete_ref, "protocol_sha256": SHA, "registration_sha256": SHA,
        "particle_identity_sha256": SHA,
        "selections": {"type": "array", "items": selection, "maxItems": 1024},
        "references": {"type": "array", "items": reference, "maxItems": 1024},
    }, ["schema_version", "status", "prepared", "template_sha256", "diagnostics"])
    result["allOf"] = [
        {"if": {"properties": {"status": {"const": "ready"}}, "required": ["status"]},
         "then": {"required": ["protocol", "protocol_sha256", "registration_sha256",
                               "particle_identity_sha256", "selections", "references"],
                  "properties": {"diagnostics": {"maxItems": 0}}}},
        {"if": {"properties": {"status": {"const": "blocked"}}, "required": ["status"]},
         "then": {"properties": {"diagnostics": {"minItems": 1}, "protocol": False,
                                 "protocol_sha256": False, "registration_sha256": False,
                                 "particle_identity_sha256": False, "selections": False,
                                 "references": False}}},
    ]
    bind = {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:neodyn:scientific-protocol-template-bind:v1",
            "title": "NeoDynamics scientific protocol template binding v1",
            "$defs": {"TemplateBindRequest": request, "TemplateBindResult": result}}
    _write("bind.schema.json", bind)


if __name__ == "__main__":
    main()
