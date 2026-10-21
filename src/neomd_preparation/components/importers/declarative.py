"""Import bounded declarative ligand assets without loading executable resources.

The caller owns persistence and authorization. This module checks exact input
bytes and returns retained asset bytes; it never opens a path or builds a System.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from uuid import NAMESPACE_URL, uuid5

import rfc8785

from neomd_preparation.components.contract import (
    ComponentContractError,
    canonical_sha256,
    imported_package_cache_key,
    validate_document,
)

_FORMAT = "neomd-declarative-ligand-import-v1"
_STRUCTURE_FORMAT = "neomd-mapped-ligand-structure-v1"
_MAX_MANIFEST = 1_048_576
_MAX_ASSET = 32 * 1024 * 1024
_MAX_ATOMS = 4_096
_NAMES = ("structure.json", "parameters.json")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class ImportedPackage:
    """Validated package and encoded assets keyed by their opaque artifact IDs."""

    package: dict
    assets: dict[str, bytes]
    source_manifest_sha256: str
    original_parameters_sha256: str


def _fail(code, path, detail):
    raise ComponentContractError(code, path, detail)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _keys(value, expected, path):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail("invalid_reference", path, "unsupported or missing declarative field")


def _decode_json(data, path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("invalid_reference", path, "duplicate JSON field")
            result[key] = value
        return result

    def nonfinite(_):
        _fail("invalid_reference", path, "nonfinite JSON number")

    try:
        value = json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)
        if rfc8785.dumps(value) != data:
            _fail("invalid_reference", path, "asset must be RFC 8785 canonical JSON")
    except ComponentContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OverflowError,
            RecursionError):
        _fail("invalid_reference", path, "invalid bounded canonical JSON")
    return value


def _preflight_members(members):
    if not isinstance(members, (list, tuple)) or len(members) != len(_NAMES):
        _fail("invalid_reference", "/assets", "exactly two declarative members are required")
    result = {}
    for index, member in enumerate(members):
        if not isinstance(member, (list, tuple)) or len(member) != 2:
            _fail("invalid_reference", f"/assets/{index}", "member needs a name and exact bytes")
        name, data = member
        if type(name) is not str or name not in _NAMES or name in result:
            _fail("invalid_reference", f"/assets/{index}/name", "unsafe, duplicate or unsupported member")
        if type(data) is not bytes:
            _fail("invalid_reference", f"/assets/{index}", "member must be exact bytes")
        if not 0 < len(data) <= _MAX_ASSET:
            _fail("oversized_payload", f"/assets/{index}", "member exceeds first import profile")
        result[name] = data
    if set(result) != set(_NAMES):
        _fail("invalid_reference", "/assets", "required declarative member is missing")
    return result


def _preflight_manifest(state, manifest_bytes, members):
    if type(manifest_bytes) is not bytes or not 0 < len(manifest_bytes) <= _MAX_MANIFEST:
        _fail("oversized_payload", "", "manifest exceeds 1 MiB or is not exact bytes")
    raw = _preflight_members(members)  # Names and byte sizes precede every JSON parse.
    manifest = _decode_json(manifest_bytes, "")
    _keys(manifest, ("format", "state_id", "state_sha256", "tool_versions",
                     "compatibility", "dependencies", "assets"), "")
    if manifest["format"] != _FORMAT or manifest["dependencies"] != []:
        _fail("invalid_reference", "/dependencies", "unsupported format or external dependency")
    if manifest["state_id"] != state["state_id"] or manifest["state_sha256"] != canonical_sha256(state):
        _fail("mapping_mismatch", "/state_sha256", "import targets a different accepted state")
    descriptors = manifest["assets"]
    if not isinstance(descriptors, list) or len(descriptors) != len(_NAMES):
        _fail("invalid_reference", "/assets", "exactly two asset descriptors are required")
    seen = set()
    for index, row in enumerate(descriptors):
        _keys(row, ("name", "sha256", "size_bytes"), f"/assets/{index}")
        name = row["name"]
        if type(name) is not str or name not in _NAMES or name in seen:
            _fail("invalid_reference", f"/assets/{index}/name", "duplicate or unsupported dependency")
        seen.add(name)
        if (type(row["size_bytes"]) is not int or row["size_bytes"] != len(raw[name])
                or type(row["sha256"]) is not str or _SHA.fullmatch(row["sha256"]) is None
                or row["sha256"] != _hash(raw[name])):
            _fail("invalid_reference", f"/assets/{index}", "declared byte identity differs")
    if seen != set(_NAMES):
        _fail("invalid_reference", "/assets", "local dependency is missing")
    versions = manifest["tool_versions"]
    if (not isinstance(versions, list) or not 1 <= len(versions) <= 32):
        _fail("invalid_reference", "/tool_versions", "versioned source declaration is required")
    tools = set()
    for index, row in enumerate(versions):
        _keys(row, ("tool", "version"), f"/tool_versions/{index}")
        if (type(row["tool"]) is not str or not 0 < len(row["tool"]) <= 128
                or type(row["version"]) is not str or not 0 < len(row["version"]) <= 128
                or row["tool"] in tools):
            _fail("invalid_reference", f"/tool_versions/{index}", "invalid or repeated tool identity")
        tools.add(row["tool"])
    return manifest, raw


def _check_structure(state, structure):
    _keys(structure, ("format", "state_id", "state_sha256", "atoms", "bonds"), "/structure")
    if (structure["format"] != _STRUCTURE_FORMAT or structure["state_id"] != state["state_id"]
            or structure["state_sha256"] != canonical_sha256(state)):
        _fail("mapping_mismatch", "/structure/state_sha256", "mapped structure targets another state")
    rows = structure["atoms"]
    if not isinstance(rows, list) or len(rows) != len(state["atoms"]):
        _fail("mapping_mismatch", "/structure/atoms", "mapped atom coverage differs")
    actual = {}
    for index, row in enumerate(rows):
        _keys(row, ("state_atom_id", "name", "element", "formal_charge", "stereo"),
              f"/structure/atoms/{index}")
        sid = row["state_atom_id"]
        if (type(sid) is not str or sid in actual or type(row["name"]) is not str
                or not 0 < len(row["name"]) <= 128):
            _fail("mapping_mismatch", f"/structure/atoms/{index}", "duplicate or unnamed mapped atom")
        actual[sid] = row
    expected = {row["id"]: row for row in state["atoms"]}
    if set(actual) != set(expected):
        _fail("mapping_mismatch", "/structure/atoms", "state atom map is incomplete")
    for sid, row in actual.items():
        atom = expected[sid]
        if (type(row["formal_charge"]) is not int
                or any(row[key] != atom[value] for key, value in
                       (("element", "element"), ("formal_charge", "formal_charge"),
                        ("stereo", "stereo")))):
            _fail("mapping_mismatch", "/structure/atoms", "mapped atom chemistry differs")
    bonds = structure["bonds"]
    if not isinstance(bonds, list) or len(bonds) != len(state["bonds"]):
        _fail("mapping_mismatch", "/structure/bonds", "mapped bond coverage differs")
    actual_bonds = set()
    for index, row in enumerate(bonds):
        _keys(row, ("atom_ids", "order", "stereo"), f"/structure/bonds/{index}")
        if (not isinstance(row["atom_ids"], list) or len(row["atom_ids"]) != 2
                or any(type(value) is not str for value in row["atom_ids"])):
            _fail("mapping_mismatch", f"/structure/bonds/{index}", "invalid bond endpoints")
        if type(row["order"]) is not str or type(row["stereo"]) is not str:
            _fail("mapping_mismatch", f"/structure/bonds/{index}", "invalid mapped bond chemistry")
        key = (tuple(row["atom_ids"]), row["order"], row["stereo"])
        if key in actual_bonds:
            _fail("mapping_mismatch", f"/structure/bonds/{index}", "duplicate mapped bond")
        actual_bonds.add(key)
    expected_bonds = {(tuple(row["atom_ids"]), row["order"], row["stereo"])
                      for row in state["bonds"]}
    if actual_bonds != expected_bonds:
        _fail("mapping_mismatch", "/structure/bonds", "mapped bond graph differs")
    return actual


def _charge_sum(parameters, state):
    if not isinstance(parameters, dict) or not isinstance(parameters.get("atoms"), list):
        _fail("invalid_reference", "/parameters/atoms", "force atom rows are required")
    rows = parameters["atoms"]
    if len(rows) != len(state["atoms"]):
        _fail("mapping_mismatch", "/parameters/atoms", "force atom coverage differs")
    state_ids = {row["id"] for row in state["atoms"]}
    seen = set()
    total = Decimal(0)
    # At most 80 characters per value and at most 80 decimal places make this
    # context wide enough to sum every supported charge without rounding.
    with localcontext() as context:
        context.prec = 100
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                _fail("invalid_reference", f"/parameters/atoms/{index}", "invalid force atom")
            sid, charge = row.get("state_atom_id"), row.get("charge_e")
            if type(sid) is not str or sid not in state_ids or sid in seen:
                _fail("mapping_mismatch", f"/parameters/atoms/{index}/state_atom_id",
                      "unknown or duplicate parameter atom")
            seen.add(sid)
            if type(charge) is not str or len(charge) > 80 or _DECIMAL.fullmatch(charge) is None:
                _fail("invalid_reference", f"/parameters/atoms/{index}/charge_e",
                      "charge must be a bounded decimal string")
            try:
                number = Decimal(charge)
            except InvalidOperation:
                _fail("invalid_reference", f"/parameters/atoms/{index}/charge_e",
                      "invalid exact charge")
            if (not number.is_finite() or abs(number.as_tuple().exponent) > 80
                    or abs(number) > 100_000):
                _fail("invalid_reference", f"/parameters/atoms/{index}/charge_e",
                      "nonfinite or unbounded exact charge")
            total += number
    return total


def _ref(kind, data, media_type="application/json"):
    digest = _hash(data)
    return {"schema_id": "urn:neodyn:component-preparation:v1:artifact-reference",
            "schema_version": 1,
            "artifact_id": str(uuid5(NAMESPACE_URL, f"neomd-component-asset:{kind}:{digest}")),
            "kind": kind, "content_sha256": digest,
            "size_bytes": len(data), "decoded_size_bytes": len(data),
            "encoding": "identity", "media_type": media_type}


def _produce(state, manifest, manifest_bytes, raw, *, adjustments=None, decision_sha256=None):
    structure = _decode_json(raw["structure.json"], "/assets/structure.json")
    names = _check_structure(state, structure)
    parameters = _decode_json(raw["parameters.json"], "/assets/parameters.json")
    _charge_sum(parameters, state)  # Check original rows before any explicit edits.
    source_parameters_bytes = raw["parameters.json"]
    original_parameters_sha256 = _hash(source_parameters_bytes)
    provenance = None
    if adjustments is not None:
        if type(decision_sha256) is not str or _SHA.fullmatch(decision_sha256) is None:
            _fail("invalid_reference", "/decision_sha256", "normalization needs exact decision evidence")
        if not isinstance(adjustments, list) or not adjustments:
            _fail("invalid_reference", "/adjustments", "explicit charge edits are required")
        if not isinstance(parameters, dict) or not isinstance(parameters.get("atoms"), list):
            _fail("invalid_reference", "/parameters/atoms", "force atom rows are required")
        parameters = copy.deepcopy(parameters)
        by_id = {row.get("state_atom_id"): row for row in parameters.get("atoms", [])
                 if isinstance(row, dict)}
        seen = set()
        changes = []
        for index, adjustment in enumerate(adjustments):
            _keys(adjustment, ("state_atom_id", "charge_e"), f"/adjustments/{index}")
            sid, charge = adjustment["state_atom_id"], adjustment["charge_e"]
            if type(sid) is not str or sid not in by_id or sid in seen or type(charge) is not str:
                _fail("mapping_mismatch", f"/adjustments/{index}", "unknown or repeated charge edit")
            seen.add(sid)
            before = by_id[sid]["charge_e"]
            if before == charge:
                _fail("invalid_reference", f"/adjustments/{index}", "charge edit has no effect")
            by_id[sid]["charge_e"] = charge
            changes.append({"state_atom_id": sid, "from": before, "to": charge})
        raw = {**raw, "parameters.json": rfc8785.dumps(parameters)}
        provenance = {"format": "neomd-import-charge-normalization-v1",
                      "original_parameters_sha256": original_parameters_sha256,
                      "decision_sha256": decision_sha256, "changes": changes}
    total = _charge_sum(parameters, state)
    residual = total - Decimal(state["net_formal_charge"])
    if abs(residual) > Decimal("0.000001"):
        _fail("mapping_mismatch", "/parameters/atoms",
              "partial charges exceed the reviewed state charge tolerance")
    parameter_ids = [row["state_atom_id"] for row in parameters["atoms"]]
    atom_map = {"format": "neomd-component-atom-map-v1",
                "atoms": [{"state_atom_id": sid, "parameter_index": index,
                           "parameter_atom_name": names[sid]["name"]}
                          for index, sid in enumerate(parameter_ids)]}
    map_bytes = rfc8785.dumps(atom_map)
    force_bytes = raw["parameters.json"]
    structure_bytes = raw["structure.json"]
    report_checks = ["Exact imported member hashes and local dependencies verified",
                     "Mapped structure and exact state graph verified",
                     f"Force coverage verified; decimal charge residual {residual} e"]
    if provenance is not None:
        report_checks.append(f"Explicit normalization from {original_parameters_sha256}")
    qc_bytes = rfc8785.dumps({"format": "neomd-component-qc-v1",
                             "status": "integrity_passed", "checks": report_checks})
    refs = [_ref("parameter_asset", structure_bytes), _ref("parameter_asset", force_bytes)]
    asset_bytes = {ref["artifact_id"]: data for ref, data in
                   zip(refs, (structure_bytes, force_bytes))}
    original_input_refs = [refs[0], refs[1]]
    if provenance is not None:
        original_ref = _ref("parameter_asset", source_parameters_bytes,
                            "application/octet-stream")
        refs.append(original_ref)
        asset_bytes[original_ref["artifact_id"]] = source_parameters_bytes
        original_input_refs = [refs[0], original_ref]
        provenance_bytes = rfc8785.dumps(provenance)
        ref = _ref("parameter_asset", provenance_bytes)
        refs.append(ref)
        asset_bytes[ref["artifact_id"]] = provenance_bytes
    map_ref, qc_ref = _ref("atom_map", map_bytes), _ref("qc_report", qc_bytes)
    asset_bytes[map_ref["artifact_id"]] = map_bytes
    asset_bytes[qc_ref["artifact_id"]] = qc_bytes
    input_hash = _hash(manifest_bytes)
    options = {"format": _FORMAT, "charge_policy": "decimal-tolerance-1e-6",
               "source_manifest_sha256": input_hash,
               "compatibility": manifest["compatibility"]}
    if provenance is not None:
        options["normalization"] = provenance
    method = {"method_id": "declarative-ligand-import", "version": "1",
              "charge_method_id": "imported-exact" if provenance is None else "imported-explicit-normalization",
              "tool_versions": manifest["tool_versions"],
              "resource_hashes": [{"resource_id": "structure.json", "sha256": _hash(structure_bytes)},
                                  {"resource_id": "parameters.json",
                                   "sha256": original_parameters_sha256}],
              "options_sha256": _hash(rfc8785.dumps(options)),
              "conformer_sha256": None, "seed": None}
    import_key = imported_package_cache_key(canonical_sha256(state), method,
                                            map_ref, original_input_refs)
    package = {"schema_id": "urn:neodyn:component-preparation:v1:parameter-package",
               "schema_version": 1,
               "package_id": str(uuid5(NAMESPACE_URL, "neomd-import:" + import_key)),
               "state_id": state["state_id"], "state_sha256": canonical_sha256(state),
               "origin": "imported", "method": method,
               "atom_map": map_ref, "assets": refs,
               "compatibility": manifest["compatibility"],
               "charge_sum_e": float(total),
               "qc": {"status": "integrity_passed", "report": qc_ref}}
    validate_document("parameter-package", package)
    # Shared scientific validator owns force vocabulary, terms and exception coverage.
    from neomd_preparation.components.assets import validate_package_assets

    validate_package_assets(state, package, asset_bytes)
    return ImportedPackage(package, asset_bytes, input_hash, original_parameters_sha256)


def import_declarative_ligand(state, manifest_bytes: bytes, members):
    """Accept exact canonical JSON members; return an integrity-checked package."""
    validate_document("chemical-state", state)
    if state["review"]["status"] != "accepted" or len(state["atoms"]) > _MAX_ATOMS:
        _fail("chemistry_unresolved", "/review/status", "accepted bounded ligand state required")
    manifest, raw = _preflight_manifest(state, manifest_bytes, members)
    return _produce(state, manifest, manifest_bytes, raw)


def normalize_imported_ligand(state, manifest_bytes: bytes, members, *,
                              charge_adjustments, decision_sha256: str):
    """Apply caller-declared charge edits to a distinct package with provenance."""
    validate_document("chemical-state", state)
    if state["review"]["status"] != "accepted" or len(state["atoms"]) > _MAX_ATOMS:
        _fail("chemistry_unresolved", "/review/status", "accepted bounded ligand state required")
    manifest, raw = _preflight_manifest(state, manifest_bytes, members)
    return _produce(state, manifest, manifest_bytes, raw,
                    adjustments=charge_adjustments, decision_sha256=decision_sha256)
