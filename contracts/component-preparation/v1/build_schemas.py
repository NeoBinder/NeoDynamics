"""Generate the self-contained component preparation v1 JSON Schemas."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "urn:neodyn:component-preparation:v1"
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
UUID = {
    "type": "string",
    "pattern": "^(?!00000000-0000-0000-0000-000000000000$)[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
}
ATOM_ID = {"type": "string", "pattern": "^[1-9][0-9]{0,19}$"}
NAME = {"type": "string", "minLength": 1, "maxLength": 128}
NOTE = {"type": "string", "minLength": 1, "maxLength": 1024}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
NONNEGATIVE = {"type": "number", "minimum": 0}


def obj(props, required=None):
    return {"type": "object", "additionalProperties": False,
            "required": list(required if required is not None else props), "properties": props}


def arr(items, maximum, minimum=0):
    return {"type": "array", "minItems": minimum, "maxItems": maximum, "items": items}


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def artifact_kind(kind):
    return {"allOf": [ref("artifact_reference"),
                      {"properties": {"kind": {"const": kind}}}]}


ARTIFACT = obj({
    "schema_id": {"const": BASE + ":artifact-reference"},
    "schema_version": {"const": 1},
    "artifact_id": UUID,
    "kind": {"enum": ["chemical_state", "component_instance", "parameter_package",
                       "parameter_asset", "atom_map", "qc_report", "preparation_recipe",
                       "prepared_bundle", "analysis"]},
    "content_sha256": SHA,
    "size_bytes": {"type": "integer", "minimum": 1, "maximum": 536870912},
    "decoded_size_bytes": {"type": "integer", "minimum": 1, "maximum": 536870912},
    "encoding": {"enum": ["identity", "gzip"]},
    "media_type": {"enum": ["application/json", "application/octet-stream", "application/gzip"]},
})

DEFS = {
    "sha256": SHA,
    "uuid": UUID,
    "atom_id": ATOM_ID,
    "artifact_reference": ARTIFACT,
    "evidence": obj({"kind": {"enum": ["ccd", "source_structure", "user_decision", "calculation", "publication"]},
                     "source_sha256": SHA, "detail": NOTE}),
    "atom": obj({"id": ATOM_ID,
                 "element": {"type": "string", "pattern": "^[A-Z][a-z]?$"},
                 "formal_charge": {"type": "integer", "minimum": -8, "maximum": 8},
                 "stereo": {"enum": ["R", "S", "none", "unresolved"]}}),
    "bond": obj({"atom_ids": {"type": "array", "prefixItems": [ATOM_ID, ATOM_ID],
                                    "minItems": 2, "maxItems": 2},
                 "order": {"enum": ["single", "double", "triple", "aromatic"]},
                 "stereo": {"enum": ["E", "Z", "none", "unresolved"]}}),
    "binding": obj({"state_atom_id": ATOM_ID, "source_atom_id": ATOM_ID}),
    "tool_version": obj({"tool": NAME, "version": NAME}),
    "resource_hash": obj({"resource_id": NAME, "sha256": SHA}),
}


def document(name, props, *, defs=None, required=None, title=None):
    schema = obj({"schema_id": {"const": BASE + ":" + name.replace("_", "-")},
                  "schema_version": {"const": 1}, **props},
                 required=["schema_id", "schema_version", *(required or props)])
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema",
                   "$id": BASE + ":" + name.replace("_", "-"),
                   "title": title or name.replace("_", " ").title(),
                   "$defs": defs or DEFS})
    filename = name.replace("_", "-") + ".schema.json"
    (ROOT / filename).write_text(json.dumps(schema, indent=2) + "\n")


document("artifact_reference", {k: v for k, v in ARTIFACT["properties"].items()
                                if k not in ("schema_id", "schema_version")},
         title="Opaque owner-scoped artifact reference")

document("chemical_state", {
    "state_id": UUID,
    "atoms": arr(ref("atom"), 100000, 1),
    "bonds": arr(ref("bond"), 200000),
    "net_formal_charge": {"type": "integer", "minimum": -100000, "maximum": 100000},
    "hydrogen_policy": {"enum": ["explicit_all", "reviewed_addition"]},
    "tautomer_id": NAME,
    "protonation_id": NAME,
    "review": obj({"status": {"enum": ["accepted", "blocked"]},
                   "decision_id": UUID, "evidence": arr(ref("evidence"), 32, 1)}),
}, title="Reviewed immutable chemical state graph")

document("component_instance", {
    "instance_id": UUID,
    "source_topology_id": UUID,
    "source_revision_id": UUID,
    "state_id": UUID,
    "state_sha256": SHA,
    "role": {"enum": ["protein", "ligand", "water", "ion", "modified_residue",
                      "covalent_group", "other"]},
    "included": {"type": "boolean"},
    "frame_id": UUID,
    "atom_bindings": arr(ref("binding"), 100000, 1),
}, title="Immutable component membership and source pose binding")

METHOD = obj({"method_id": NAME, "version": NAME, "charge_method_id": NAME,
              "tool_versions": arr(ref("tool_version"), 32, 1),
              "resource_hashes": arr(ref("resource_hash"), 64, 1),
              "options_sha256": SHA,
              "conformer_sha256": {"anyOf": [SHA, {"type": "null"}]},
              "seed": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 4294967295},
                                {"type": "null"}]}})
COMPATIBILITY = obj({
    "protein_family": NAME, "ligand_family": NAME, "water_family": NAME,
    "ion_family": NAME, "combining_rule": NAME,
    "electrostatic_14_scale": POSITIVE, "vdw_14_scale": POSITIVE,
    "unit_system": {"const": "openmm_nm_kj_mol_e"},
    "nonbonded_method": {"enum": ["pme", "cutoff_periodic", "no_cutoff"]},
    "cutoff_nm": POSITIVE,
    "switch_distance_nm": {"anyOf": [NONNEGATIVE, {"type": "null"}]},
    "constraints": {"enum": ["none", "HBonds", "AllBonds", "HAngles"]},
    "hydrogen_mass_amu": POSITIVE,
    "virtual_sites": {"type": "boolean"},
})
DEFS.update({"method": METHOD, "compatibility": COMPATIBILITY})

document("parameter_package", {
    "package_id": UUID, "state_id": UUID, "state_sha256": SHA,
    "origin": {"enum": ["generated", "imported"]},
    "method": ref("method"),
    "atom_map": artifact_kind("atom_map"),
    "assets": arr(artifact_kind("parameter_asset"), 64, 1),
    "compatibility": ref("compatibility"),
    "charge_sum_e": {"type": "number"},
    "qc": obj({"status": {"enum": ["unreviewed", "integrity_passed",
                                   "numerically_executable", "scientifically_qualified", "blocked"]},
               "report": artifact_kind("qc_report")}),
    "attachment_sites": arr(obj({"state_atom_id": ATOM_ID,
                                  "partner_instance_id": UUID,
                                  "partner_state_atom_id": ATOM_ID,
                                  "interface_terms": artifact_kind("parameter_asset")}), 16),
}, required=["package_id", "state_id", "state_sha256", "origin", "method", "atom_map",
             "assets", "compatibility", "charge_sum_e", "qc"],
         title="State-bound immutable parameter package manifest")

SOURCE_TOPOLOGY = obj({
    "schema_id": {"const": "urn:neodyn:molecular-topology:v1"},
    "topology_id": UUID, "revision_id": UUID, "content_sha256": SHA,
    "size_bytes": {"type": "integer", "minimum": 1, "maximum": 67108864},
    "decoded_size_bytes": {"type": "integer", "minimum": 1, "maximum": 268435456},
    "encoding": {"const": "gzip"},
})
ASSIGNMENT = obj({"instance_id": UUID, "instance_sha256": SHA,
                  "state_id": UUID, "state_sha256": SHA,
                  "package_id": UUID, "package_sha256": SHA})
ASSEMBLY = obj({
    "box_policy": {"enum": ["reuse_source_orthorhombic", "generate_orthorhombic"]},
    "padding_nm": POSITIVE,
    "retained_water_policy": {"enum": ["retain_and_solvate", "discard_source_water"]},
    "water_model": NAME,
    "ionic_strength_mol_per_l": NONNEGATIVE,
    "neutralize": {"type": "boolean"},
    "solvation_seed": {"type": "integer", "minimum": 0, "maximum": 4294967295},
    "positive_ion": NAME, "negative_ion": NAME,
    "constraints": COMPATIBILITY["properties"]["constraints"],
    "hydrogen_mass_amu": POSITIVE,
    "nonbonded_method": COMPATIBILITY["properties"]["nonbonded_method"],
    "cutoff_nm": POSITIVE,
    "switch_distance_nm": {"anyOf": [NONNEGATIVE, {"type": "null"}]},
})
DEFS.update({"source_topology": SOURCE_TOPOLOGY, "assignment": ASSIGNMENT,
             "assembly": ASSEMBLY})

document("preparation_recipe", {
    "recipe_id": UUID,
    "source_topology": ref("source_topology"),
    "assignments": arr(ref("assignment"), 4096, 1),
    "assembly": ref("assembly"),
}, title="Immutable component preparation and assembly recipe")

QUALIFICATION = obj({"source_sha256": SHA, "resource_sha256": SHA,
                     "platform": NAME, "chemistry_scope": NAME, "receipt_sha256": SHA})
CATALOG_METHOD = obj({
    "method_id": NAME, "version": NAME,
    "chemistry_scope": arr(NAME, 32, 1),
    "import_mode": {"enum": ["generated", "imported"]},
    "resource_hashes": arr(ref("resource_hash"), 64, 1),
    "availability": {"enum": ["qualified", "unavailable"]},
    "qualification": {"anyOf": [QUALIFICATION, {"type": "null"}]},
    "reason": {"anyOf": [NOTE, {"type": "null"}]},
})
CATALOG_METHOD["allOf"] = [
    {"if": {"properties": {"availability": {"const": "qualified"}}},
     "then": {"properties": {"qualification": QUALIFICATION, "reason": {"type": "null"}}}},
    {"if": {"properties": {"availability": {"const": "unavailable"}}},
     "then": {"properties": {"qualification": {"type": "null"}, "reason": NOTE}}},
]
DEFS.update({"catalog_method": CATALOG_METHOD})
document("capability_catalog", {
    "catalog_revision_sha256": SHA,
    "methods": arr(ref("catalog_method"), 128),
}, title="Qualified component preparation capability catalog")

document("diagnostic", {
    "code": {"enum": ["contract_version_mismatch", "invalid_reference", "oversized_payload",
                      "mapping_mismatch", "chemistry_unresolved", "incompatible_package",
                      "capability_unavailable"]},
    "path": {"type": "string", "pattern": "^(|/.*)$", "maxLength": 1024},
    "detail": NOTE,
}, title="Bounded structured component preparation diagnostic")
