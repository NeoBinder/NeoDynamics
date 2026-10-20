"""Validate component preparation v1 artifacts and their scientific links.

Schemas define wire shape; this module checks graph, charge and mapping facts
that cannot be expressed with local JSON Schema constraints.
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib

import rfc8785
from jsonschema import Draft202012Validator

from .contracts import schema_dir

BASE = "urn:neodyn:component-preparation:v1"
_SCHEMA_DIR = schema_dir("component-preparation/v1")
_KINDS = ("artifact-reference", "chemical-state", "component-instance",
          "parameter-package", "preparation-recipe", "capability-catalog", "diagnostic")
_VALIDATORS = {
    kind: Draft202012Validator(json.loads((_SCHEMA_DIR / f"{kind}.schema.json").read_text()))
    for kind in _KINDS
}
_ARTIFACT_LIMITS = {
    "chemical_state": (268435456, 268435456),
    "component_instance": (268435456, 268435456),
    "parameter_package": (1048576, 1048576),
    "preparation_recipe": (1048576, 1048576),
    "parameter_asset": (536870912, 536870912),
    "atom_map": (536870912, 536870912),
    "qc_report": (1048576, 1048576),
    "prepared_bundle": (536870912, 536870912),
    "analysis": (536870912, 536870912),
}


class ComponentContractError(ValueError):
    """Bounded contract failure with an RFC 6901 JSON Pointer."""

    def __init__(self, code: str, path: str, detail: str):
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code} at {path or '/'}: {detail}")


def _fail(code, path, detail):
    raise ComponentContractError(code, path, detail)


def _pointer(parts):
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _atom_number(value, path):
    number = int(value)
    if number > (1 << 64) - 1:
        _fail("mapping_mismatch", path, "atom ID exceeds u64 range")
    return number


def canonical_sha256(document):
    """Hash decoded JSON by RFC 8785, independent of encoded transfer bytes."""
    return hashlib.sha256(rfc8785.dumps(document)).hexdigest()


def generated_package_cache_key(state_sha256: str, method: dict):
    """Bind every generation input named by the frozen v1 cache policy.

    The source pose/instance is deliberately absent.  A charge choice belongs
    in options_sha256; callers must hash its exact canonical options document.
    """
    if not isinstance(state_sha256, str) or len(state_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in state_sha256
    ):
        _fail("invalid_reference", "/state_sha256", "invalid state hash")
    definitions = _VALIDATORS["parameter-package"].schema["$defs"]
    validator = Draft202012Validator({"$defs": definitions, **definitions["method"]})
    if next(validator.iter_errors(method), None) is not None:
        _fail("invalid_reference", "/method", "invalid generation method inputs")
    tools = {row["tool"]: row["version"] for row in method["tool_versions"]}
    resources = {row["resource_id"]: row["sha256"] for row in method["resource_hashes"]}
    if len(tools) != len(method["tool_versions"]) or len(resources) != len(method["resource_hashes"]):
        _fail("invalid_reference", "/method", "duplicate tool or resource identity")
    inputs = ["component-package-cache-v1", state_sha256, method["method_id"],
              method["version"], method["charge_method_id"], tools,
              [[key, resources[key]] for key in sorted(resources)], method["options_sha256"],
              method["conformer_sha256"], method["seed"]]
    return hashlib.sha256(rfc8785.dumps(inputs)).hexdigest()


def imported_package_cache_key(state_sha256: str, method: dict, atom_map_ref: dict,
                               input_asset_refs: list[dict]):
    """Bind an import to exact original parameter/template bytes and policy."""
    if not isinstance(state_sha256, str) or len(state_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in state_sha256
    ):
        _fail("invalid_reference", "/state_sha256", "invalid state hash")
    definitions = _VALIDATORS["parameter-package"].schema["$defs"]
    validator = Draft202012Validator({"$defs": definitions, **definitions["method"]})
    if next(validator.iter_errors(method), None) is not None:
        _fail("invalid_reference", "/method", "invalid import method inputs")
    validate_document("artifact-reference", atom_map_ref)
    if atom_map_ref["kind"] != "atom_map" or not input_asset_refs:
        _fail("invalid_reference", "/atom_map", "import needs its mapped structure and input assets")
    seen = set()
    inputs = []
    for index, ref in enumerate(input_asset_refs):
        validate_document("artifact-reference", ref)
        key = (ref["kind"], ref["artifact_id"])
        if key in seen or ref["kind"] != "parameter_asset":
            _fail("invalid_reference", f"/input_assets/{index}", "duplicate or nonparameter input asset")
        seen.add(key)
        inputs.append([ref["kind"], ref["artifact_id"], ref["content_sha256"]])
    value = ["component-package-import-v1", state_sha256, method["method_id"],
             method["version"], atom_map_ref["content_sha256"], sorted(inputs),
             method["options_sha256"]]
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def validate_document(kind: str, document: dict):
    """Check a versioned document and local invariants; return the same value."""
    if kind not in _VALIDATORS:
        _fail("contract_version_mismatch", "/schema_id", "unknown contract kind")
    if not isinstance(document, dict):
        _fail("contract_version_mismatch", "", "document must be an object")
    if document.get("schema_id") != BASE + ":" + kind or document.get("schema_version") != 1:
        _fail("contract_version_mismatch", "/schema_id", "unsupported schema identity or version")
    if (kind == "artifact-reference" and document.get("kind") in _ARTIFACT_LIMITS
            and isinstance(document.get("size_bytes"), int)
            and isinstance(document.get("decoded_size_bytes"), int)):
        encoded_max, decoded_max = _ARTIFACT_LIMITS[document["kind"]]
        if document["size_bytes"] > encoded_max or document["decoded_size_bytes"] > decoded_max:
            _fail("oversized_payload", "/size_bytes", "artifact exceeds kind-specific limit")
    error = next(iter(_VALIDATORS[kind].iter_errors(document)), None)
    if error is not None:
        _fail("invalid_reference", _pointer(error.absolute_path), "document does not match local schema")
    if kind in ("parameter-package", "preparation-recipe", "capability-catalog", "diagnostic"):
        try:
            encoded_length = len(rfc8785.dumps(document))
        except (TypeError, ValueError, OverflowError):
            _fail("invalid_reference", "", "control JSON is not finite canonical JSON")
        if encoded_length > 1048576:
            _fail("oversized_payload", "", "control JSON exceeds 1 MiB")
    if kind == "artifact-reference":
        _validate_artifact_reference(document)
    elif kind == "chemical-state":
        _validate_state(document)
    elif kind == "component-instance":
        _validate_instance(document)
    elif kind == "parameter-package":
        _validate_package(document)
    elif kind == "preparation-recipe":
        _validate_recipe(document)
    elif kind == "capability-catalog":
        _validate_catalog(document)
    elif kind == "diagnostic":
        _validate_pointer(document["path"])
    return document


def _validate_pointer(pointer):
    # RFC 6901 permits only ~0 and ~1 as tilde escapes.
    for index, char in enumerate(pointer):
        if char == "~" and (index + 1 == len(pointer) or pointer[index + 1] not in "01"):
            _fail("invalid_reference", "/path", "invalid JSON Pointer escape")


def _validate_artifact_reference(ref):
    encoded, decoded = _ARTIFACT_LIMITS[ref["kind"]]
    if ref["size_bytes"] > encoded or ref["decoded_size_bytes"] > decoded:
        _fail("oversized_payload", "/size_bytes", "artifact exceeds kind-specific limit")
    if ref["encoding"] == "identity":
        if ref["size_bytes"] != ref["decoded_size_bytes"]:
            _fail("invalid_reference", "/decoded_size_bytes", "identity lengths differ")
        if ref["media_type"] == "application/gzip":
            _fail("invalid_reference", "/media_type", "identity artifact has gzip media type")
    elif ref["media_type"] != "application/gzip":
        _fail("invalid_reference", "/media_type", "gzip artifact needs gzip media type")


def _validate_state(state):
    atoms = state["atoms"]
    ids = [_atom_number(atom["id"], f"/atoms/{index}/id")
           for index, atom in enumerate(atoms)]
    if ids != sorted(set(ids)):
        _fail("mapping_mismatch", "/atoms", "state atom IDs must be unique and sorted")
    if sum(atom["formal_charge"] for atom in atoms) != state["net_formal_charge"]:
        _fail("mapping_mismatch", "/net_formal_charge", "formal atom charges do not sum")
    seen = set()
    for index, bond in enumerate(state["bonds"]):
        a, b = (_atom_number(value, f"/bonds/{index}/atom_ids/{endpoint}")
                for endpoint, value in enumerate(bond["atom_ids"]))
        if a >= b or a not in ids or b not in ids or (a, b) in seen:
            _fail("mapping_mismatch", f"/bonds/{index}/atom_ids", "invalid or duplicate bond pair")
        seen.add((a, b))
    if state["review"]["status"] == "accepted":
        if any(atom["stereo"] == "unresolved" for atom in atoms) or any(
            bond["stereo"] == "unresolved" for bond in state["bonds"]
        ) or any(state[key].lower() == "unresolved" for key in ("tautomer_id", "protonation_id")):
            _fail("chemistry_unresolved", "/review/status", "accepted chemistry is unresolved")


def _validate_instance(instance):
    state_ids = [row["state_atom_id"] for row in instance["atom_bindings"]]
    source_ids = [row["source_atom_id"] for row in instance["atom_bindings"]]
    for index, row in enumerate(instance["atom_bindings"]):
        for field in ("state_atom_id", "source_atom_id"):
            _atom_number(row[field], f"/atom_bindings/{index}/{field}")
    if len(set(state_ids)) != len(state_ids) or len(set(source_ids)) != len(source_ids):
        _fail("mapping_mismatch", "/atom_bindings", "atom bindings must be one-to-one")
    if list(map(int, state_ids)) != sorted(map(int, state_ids)):
        _fail("mapping_mismatch", "/atom_bindings", "bindings must be ordered by state atom ID")


def _validate_package(package):
    for key in ("atom_map", "qc"):
        ref = package[key] if key == "atom_map" else package[key]["report"]
        validate_document("artifact-reference", ref)
    for ref in package["assets"]:
        validate_document("artifact-reference", ref)
    for site in package.get("attachment_sites", []):
        validate_document("artifact-reference", site["interface_terms"])
    if not math.isfinite(package["charge_sum_e"]):
        _fail("mapping_mismatch", "/charge_sum_e", "charge sum must be finite")
    if len({row["resource_id"] for row in package["method"]["resource_hashes"]}) != len(
        package["method"]["resource_hashes"]
    ):
        _fail("invalid_reference", "/method/resource_hashes", "duplicate resource ID")


def _validate_recipe(recipe):
    ids = [row["instance_id"] for row in recipe["assignments"]]
    if len(ids) != len(set(ids)):
        _fail("mapping_mismatch", "/assignments", "duplicate component assignment")
    assembly = recipe["assembly"]
    if (assembly["switch_distance_nm"] is not None
            and assembly["switch_distance_nm"] >= assembly["cutoff_nm"]):
        _fail("incompatible_package", "/assembly/switch_distance_nm", "switch must precede cutoff")


def _validate_catalog(catalog):
    keys = [(row["method_id"], row["version"], row["import_mode"])
            for row in catalog["methods"]]
    if len(keys) != len(set(keys)):
        _fail("invalid_reference", "/methods", "duplicate catalog method")


def validate_component_set(states, instances, packages, recipe, *, source_atom_ids):
    """Resolve a recipe against supplied immutable records and canonical hashes.

    ``source_atom_ids`` maps instance UUIDs to *all* source-present atom IDs in
    that component, derived from the authoritative molecular-topology v1 graph.
    Generated hydrogens have no source binding and may be absent from bindings.
    """
    validate_document("preparation-recipe", recipe)
    by_state = {}
    for state in states:
        validate_document("chemical-state", state)
        if state["state_id"] in by_state:
            _fail("mapping_mismatch", "/states", "duplicate state identity")
        by_state[state["state_id"]] = state
    by_package = {}
    for package in packages:
        validate_document("parameter-package", package)
        if package["package_id"] in by_package:
            _fail("mapping_mismatch", "/packages", "duplicate package identity")
        by_package[package["package_id"]] = package
        state = by_state.get(package["state_id"])
        if state is None or package["state_sha256"] != canonical_sha256(state):
            _fail("mapping_mismatch", "/state_sha256", "package state identity differs")
        if abs(package["charge_sum_e"] - state["net_formal_charge"]) > 1e-6:
            _fail("mapping_mismatch", "/charge_sum_e", "package charge sum differs from state")
    by_instance = {}
    all_source_ids = set()
    for instance in instances:
        validate_document("component-instance", instance)
        iid = instance["instance_id"]
        if iid in by_instance:
            _fail("mapping_mismatch", "/instances", "duplicate instance identity")
        by_instance[iid] = instance
        state = by_state.get(instance["state_id"])
        if state is None or instance["state_sha256"] != canonical_sha256(state):
            _fail("mapping_mismatch", "/state_sha256", "instance state identity differs")
        atom_ids = {row["id"] for row in state["atoms"]}
        mapped = {row["state_atom_id"] for row in instance["atom_bindings"]}
        if not mapped <= atom_ids:
            _fail("mapping_mismatch", "/atom_bindings", "binding refers to unknown state atom")
        expected = set(source_atom_ids.get(iid, []))
        actual = {row["source_atom_id"] for row in instance["atom_bindings"]}
        if actual != expected:
            _fail("mapping_mismatch", "/atom_bindings", "source atom binding is incomplete")
        if all_source_ids & actual:
            _fail("mapping_mismatch", "/atom_bindings", "source atom belongs to multiple instances")
        all_source_ids.update(actual)
        if (instance["source_topology_id"] != recipe["source_topology"]["topology_id"]
                or instance["source_revision_id"] != recipe["source_topology"]["revision_id"]):
            _fail("mapping_mismatch", "/source_topology", "instance source revision differs")
    assigned = {row["instance_id"] for row in recipe["assignments"]}
    if assigned != {iid for iid, instance in by_instance.items() if instance["included"]}:
        _fail("mapping_mismatch", "/assignments", "assignments must cover included instances once")
    for index, row in enumerate(recipe["assignments"]):
        instance = by_instance[row["instance_id"]]
        state = by_state[instance["state_id"]]
        package = by_package.get(row["package_id"])
        if package is None:
            _fail("mapping_mismatch", f"/assignments/{index}/package_id", "package is missing")
        if state["review"]["status"] != "accepted":
            _fail("chemistry_unresolved", f"/assignments/{index}/state_id", "state is not accepted")
        if (row["instance_sha256"] != canonical_sha256(instance)
                or row["state_id"] != state["state_id"]
                or row["state_sha256"] != canonical_sha256(state)
                or package["state_id"] != state["state_id"]
                or row["package_sha256"] != canonical_sha256(package)):
            _fail("mapping_mismatch", f"/assignments/{index}", "assignment hash or identity differs")
        if not instance["included"]:
            _fail("mapping_mismatch", f"/assignments/{index}/instance_id", "excluded instance assigned")
    return recipe


def verify_artifact_bytes(ref, encoded: bytes):
    """Verify encoded identity before bounded decompression; return decoded bytes."""
    validate_document("artifact-reference", ref)
    if len(encoded) != ref["size_bytes"] or hashlib.sha256(encoded).hexdigest() != ref["content_sha256"]:
        _fail("invalid_reference", "/content_sha256", "encoded artifact identity differs")
    if ref["encoding"] == "identity":
        return encoded
    decoder = zlib.decompressobj(wbits=31)
    decoded = bytearray()
    try:
        for offset in range(0, len(encoded), 65536):
            part = decoder.decompress(encoded[offset:offset + 65536],
                                      ref["decoded_size_bytes"] + 1 - len(decoded))
            decoded.extend(part)
            if len(decoded) > ref["decoded_size_bytes"] or decoder.unconsumed_tail:
                _fail("oversized_payload", "/decoded_size_bytes", "decoded artifact exceeds bound")
    except zlib.error:
        _fail("invalid_reference", "/encoding", "invalid gzip artifact")
    if not decoder.eof or decoder.unused_data or len(decoded) != ref["decoded_size_bytes"]:
        _fail("invalid_reference", "/decoded_size_bytes", "gzip member is incomplete or has trailing bytes")
    return bytes(decoded)


def decode_document_artifact(ref, encoded: bytes, kind: str):
    """Decode a canonical JSON artifact with strict keys, schema and invariants."""
    if kind not in ("chemical-state", "component-instance", "parameter-package",
                    "preparation-recipe") or ref.get("kind") != kind.replace("-", "_"):
        _fail("invalid_reference", "/kind", "artifact kind and document kind differ")
    decoded = verify_artifact_bytes(ref, encoded)
    if ref["encoding"] == "identity" and ref["media_type"] != "application/json":
        _fail("invalid_reference", "/media_type", "JSON artifact needs JSON media type")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                _fail("invalid_reference", "", "duplicate JSON key")
            value[key] = item
        return value

    def nonfinite(_):
        _fail("invalid_reference", "", "nonfinite JSON number")

    try:
        document = json.loads(decoded, object_pairs_hook=unique, parse_constant=nonfinite)
        if rfc8785.dumps(document) != decoded:
            _fail("invalid_reference", "", "artifact JSON is not RFC 8785 canonical")
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError):
        _fail("invalid_reference", "", "invalid canonical JSON artifact")
    return validate_document(kind, document)
