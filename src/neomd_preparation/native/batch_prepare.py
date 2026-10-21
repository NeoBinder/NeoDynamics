"""Trusted frozen-input transport for native preparation (science plugs in separately)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .native_preparation import CONTRACT, PROFILE, contract_directory, inspect_native_preparation

MAX_REQUEST = 65536
MAX_RESULT = 65536
MAX_MEMBER = 268435456
MAX_TOTAL = 536870912
MAX_IDENTITY = 16777216
TOKEN = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
REQUEST_KEYS = set("schema_version config_contract config_contract_sha256 cwd declared_outputs document_reference execution_profile executor_binding file_bindings materializer_version native_config_sha256 native_config_yaml native_typed_sha256 prepare_yaml_sha256 prepare_yaml_utf8 profile_revision scientific_input_sha256 source_box_policy source_manifest source_manifest_sha256 typed_comparison".split())


class BatchPreparationError(ValueError):
    """A safe transport or scientific failure with no source/private path disclosure."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(condition, code="binding_invalid"):
    if not condition:
        raise BatchPreparationError(code)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")



def serialize_terminal_result(result):
    """Bound canonical UTF-8 bytes by removing only displayed QC issue suffixes."""
    data = canonical_json(result)
    if len(data) <= MAX_RESULT:
        return result, data
    bounded = copy.deepcopy(result)
    checks = (bounded.get("qc") or {}).get("checks", [])
    while len(data) > MAX_RESULT:
        candidates = [check for check in checks if check["issues"]]
        if not candidates:
            raise BatchPreparationError("result_size_exceeded")
        # Drop the largest trailing issue, preserving each check's ordered prefix.
        check = max(candidates, key=lambda check: len(canonical_json(check["issues"][-1])))
        check["issues"].pop()
        check["displayed_findings"] = len(check["issues"])
        check["truncated"] = check["total_findings"] > len(check["issues"])
        data = canonical_json(bounded)
    return bounded, data


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def pairs(entries):
        result = {}
        for key, value in entries:
            _require(key not in result, "archive_invalid")
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise BatchPreparationError("archive_invalid") from None


def _closed(value, keys):
    _require(isinstance(value, dict) and set(value) == set(keys))


def _uuid(value):
    try:
        _require(isinstance(value, str) and str(UUID(value)) == value)
    except (ValueError, AttributeError):
        raise BatchPreparationError("source_identity_mismatch") from None


def _valid_path(name):
    return (isinstance(name, str) and len(name) <= 256 and "\\" not in name
            and all(ord(c) >= 32 for c in name)
            and all(part not in {"", ".", ".."} for part in name.split("/")))


@dataclass(frozen=True)
class MaterializedPreparation:
    request: dict
    config: dict
    work_dir: Path
    execution_yaml: Path
    source_identities: dict


def materialize_preparation(request_path, source_bundle, work_dir="/work"):
    """Validate every byte before writing; preserve both YAMLs exactly, without defaults."""
    with open(request_path, "rb") as handle:
        raw = handle.read(MAX_REQUEST + 1)
    _require(len(raw) <= MAX_REQUEST, "request_size_exceeded")
    request = _json(raw)
    _closed(request, REQUEST_KEYS)
    _require(request["schema_version"] == "neomd.materialized-preparation/v1")
    _require(request["config_contract"] == CONTRACT, "config_contract_mismatch")
    _require(request["execution_profile"] == PROFILE and type(request["profile_revision"]) is int
             and request["profile_revision"] == 1, "profile_unsupported")
    _require(request["cwd"] == "/work" and request["declared_outputs"] == ["result.json", "artifacts.zip"])
    _require(request["materializer_version"] == "neoapi.md-materializer/v1"
             and request["source_box_policy"] == "require_usable_source_periodic_box")
    binding = request["executor_binding"]
    if "binding_scope" in binding:
        _closed(binding, {"git_revision", "source_snapshot_sha256", "binding_scope"})
        _require(binding["binding_scope"] == "local_source")
        _require(isinstance(binding["source_snapshot_sha256"], str)
                 and HASH.fullmatch(binding["source_snapshot_sha256"]) is not None)
    else:
        _closed(binding, {"git_revision", "image_digest"})
        _require(re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_digest"]) is not None)
    _require(re.fullmatch(r"[0-9a-f]{40}", binding["git_revision"]) is not None)
    _require(isinstance(request["config_contract_sha256"], str)
             and request["config_contract_sha256"] == sha256((contract_directory() / "native-fields.json").read_bytes()), "config_contract_mismatch")
    if request["document_reference"] is not None:
        _closed(request["document_reference"], {"document_id", "revision"})
        _uuid(request["document_reference"]["document_id"])
        _require(type(request["document_reference"]["revision"]) is int and request["document_reference"]["revision"] >= 1)
    inspections = []
    for text_key, hash_key in [("native_config_yaml", "native_config_sha256"), ("prepare_yaml_utf8", "prepare_yaml_sha256")]:
        _require(isinstance(request[text_key], str))
        inspection = inspect_native_preparation(request[text_key], execution_profile=PROFILE)
        _require(inspection.valid, inspection.issues[0]["code"] if inspection.issues else "native_invalid_yaml")
        _require(inspection.yaml_sha256 == request[hash_key], "manifest_hash_mismatch")
        inspections.append(inspection)
    native, execution = inspections
    _require(native.native_typed_sha256 == request["native_typed_sha256"], "manifest_hash_mismatch")
    manifest = request["source_manifest"]
    _closed(manifest, {"schema_version", "entries"})
    _require(manifest["schema_version"] == "neomd.source-manifest/v1")
    entries = manifest["entries"]
    _require(isinstance(entries, list) and 0 < len(entries) <= 128, "manifest_limit_exceeded")
    by_id, by_path = {}, {}
    for entry in entries:
        _closed(entry, {"id", "member_path", "format", "size_bytes", "sha256"})
        ident, path, fmt = entry["id"], entry["member_path"], entry["format"]
        _require(isinstance(ident, str) and TOKEN.fullmatch(ident) is not None)
        _require(_valid_path(path) and ident not in by_id and path not in by_path)
        _require(fmt in {"pdb", "pdbx", "sdf", "mol2", "mol", "source-instance-json"})
        _require(type(entry["size_bytes"]) is int and 0 <= entry["size_bytes"] <= MAX_MEMBER, "archive_limit_exceeded")
        _require(isinstance(entry["sha256"], str) and HASH.fullmatch(entry["sha256"]) is not None)
        if fmt != "source-instance-json":
            _require(path == f"assets/{ident}.{fmt}")
        by_id[ident], by_path[path] = entry, entry
    _require(list(by_id) == sorted(by_id))
    canonical = canonical_json(manifest)
    _require(sha256(canonical) == request["source_manifest_sha256"], "manifest_hash_mismatch")
    bindings = request["file_bindings"]
    _require(isinstance(bindings, list) and len(bindings) <= 64)
    expected = [("protein", "path")] + [("ligands", key, "path") for key in native.data.get("ligands", {})]
    seen, used, instances, sources, identity_entries = set(), set(), set(), [], {}
    rewritten = copy.deepcopy(native.typed_tree)
    changed = []
    def replace(tree, path, value):
        if not path:
            _require(tree[0] == "string")
            tree[1] = value
            return
        _require(tree[0] == "map")
        for key, child in tree[1]:
            if key == ["string", path[0]]:
                return replace(child, path[1:], value)
        raise BatchPreparationError("binding_invalid")
    for binding in bindings:
        _closed(binding, {"config_path", "source_id", "source_instance_id", "identity_source_id"})
        path = binding["config_path"]
        _require(isinstance(path, list) and all(isinstance(p, str) for p in path))
        path = tuple(path)
        _require(path in expected and path not in seen)
        seen.add(path)
        _uuid(binding["source_instance_id"])
        _require(binding["source_instance_id"] not in instances, "source_identity_mismatch")
        instances.add(binding["source_instance_id"])
        source, identity = by_id.get(binding["source_id"]), by_id.get(binding["identity_source_id"])
        _require(source is not None and identity is not None)
        _require(source["id"] not in used and identity["id"] not in used)
        used.update((source["id"], identity["id"]))
        _require(source["format"] in ({"pdb", "pdbx"} if path[0] == "protein" else {"pdb", "sdf", "mol2", "mol"}))
        _require(identity["format"] == "source-instance-json"
                 and identity["member_path"] == f"identities/{source['id']}.json")
        replace(rewritten, path, "/work/" + source["member_path"])
        changed.append(list(path))
        sources.append({"config_path": list(path), "source_instance_id": binding["source_instance_id"],
                        "content_sha256": source["sha256"], "identity_sha256": identity["sha256"]})
        identity_entries[identity["member_path"]] = (binding, source)
    _require(seen == set(expected) and used == set(by_id))
    replace(rewritten, ("output_dir",), "/work/output")
    changed.append(["output_dir"])
    _require(rewritten == execution.typed_tree, "binding_invalid")
    _require(request["typed_comparison"] == {"changed_paths": changed, "only_declared_paths_changed": True})
    scientific = {"typed_config": native.typed_tree, "config_contract": CONTRACT, "execution_profile": PROFILE,
                  "profile_revision": 1, "binding_state": "bound", "sources": sorted(sources, key=lambda x: x["config_path"])}
    _require(sha256(canonical_json(scientific)) == request["scientific_input_sha256"], "manifest_hash_mismatch")
    _require(Path(source_bundle).stat().st_size <= MAX_MEMBER, "archive_limit_exceeded")
    payloads, identities, total = {}, {}, 0
    try:
        with zipfile.ZipFile(source_bundle) as archive:
            infos = archive.infolist()
            _require(len(infos) == len(entries) + 1 and len(infos) <= 129, "archive_invalid")
            names = [info.filename for info in infos]
            _require(len(set(n.casefold() for n in names)) == len(names)
                     and set(names) == set(by_path) | {"manifest.json"}, "archive_invalid")
            for info in infos:
                mode = info.external_attr >> 16
                _require(_valid_path(info.filename) and not info.is_dir() and not info.flag_bits & 1
                         and (stat.S_IFMT(mode) in {0, stat.S_IFREG})
                         and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}, "archive_invalid")
                limit = info.file_size if info.filename == "manifest.json" else by_path[info.filename]["size_bytes"]
                if info.filename == "manifest.json":
                    _require(limit <= MAX_REQUEST, "archive_limit_exceeded")
                if info.filename in identity_entries:
                    _require(limit <= MAX_IDENTITY, "archive_limit_exceeded")
                _require(info.file_size == limit and info.file_size <= MAX_MEMBER
                         and info.file_size <= max(1, info.compress_size) * 100, "archive_limit_exceeded")
                chunks, size = [], 0
                with archive.open(info) as handle:
                    while chunk := handle.read(min(1048576, limit - size + 1)):
                        size += len(chunk)
                        total += len(chunk)
                        _require(size <= limit and total <= MAX_TOTAL, "archive_limit_exceeded")
                        chunks.append(chunk)
                data = b"".join(chunks)
                _require(size == limit, "archive_invalid")
                if info.filename == "manifest.json":
                    _require(canonical_json(_json(data)) == canonical, "manifest_hash_mismatch")
                else:
                    _require(sha256(data) == by_path[info.filename]["sha256"], "manifest_hash_mismatch")
                    payloads[info.filename] = data
    except (zipfile.BadZipFile, RuntimeError, EOFError, NotImplementedError):
        raise BatchPreparationError("archive_invalid") from None
    global_atoms = set()
    for name, (binding, source) in identity_entries.items():
        identity = _json(payloads[name])
        _validate_identity(identity, binding, source, native.data)
        atoms = {atom["source_atom_id"] for atom in identity.get("atom_bindings", [])}
        _require(not atoms & global_atoms, "source_identity_mismatch")
        global_atoms.update(atoms)
        identities[binding["source_instance_id"]] = identity
    root = Path(work_dir).absolute()
    _require(not root.is_symlink() and (not root.exists() or not any(root.iterdir())), "binding_invalid")
    root.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (root / "native-config.yaml").write_bytes(request["native_config_yaml"].encode("utf-8"))
    yaml_path = root / "prepare.yaml"
    yaml_path.write_bytes(request["prepare_yaml_utf8"].encode("utf-8"))
    return MaterializedPreparation(request, execution.data, root, yaml_path, identities)


def _validate_identity(identity, binding, source, config):
    base = {"schema_version", "binding_state", "source_instance_id", "content_sha256", "input_key", "kind"}
    _require(isinstance(identity, dict), "source_identity_mismatch")
    state = identity.get("binding_state")
    _require(state in {"new_instance", "verified_reuse"}, "source_identity_mismatch")
    _closed(identity, base | ({"loader", "atom_bindings"} if state == "verified_reuse" else set()))
    path = binding["config_path"]
    key = "protein" if path[0] == "protein" else f"ligand:{list(config.get('ligands', {})).index(path[1])}"
    _require(identity["schema_version"] == "neomd.source-instance/v1"
             and identity["source_instance_id"] == binding["source_instance_id"]
             and identity["content_sha256"] == source["sha256"] and identity["input_key"] == key
             and identity["kind"] == ("structure" if key == "protein" else "ligand"), "source_identity_mismatch")
    if state == "new_instance":
        return
    _closed(identity["loader"], {"name", "version", "atom_order"})
    loader = identity["loader"]
    _require(loader["atom_order"] == "preparation_input" and all(isinstance(loader[k], str) and 0 < len(loader[k]) <= 256 for k in ("name", "version")), "source_identity_mismatch")
    atoms = identity["atom_bindings"]
    _require(isinstance(atoms, list) and 0 < len(atoms) <= 500000, "source_identity_mismatch")
    indices, atom_ids = set(), set()
    for atom in atoms:
        _closed(atom, {"input_atom_index", "source_atom_id", "source_residue_id", "element", "position_nm"})
        index = atom["input_atom_index"]
        _require(type(index) is int and 0 <= index < len(atoms) and index not in indices, "source_identity_mismatch")
        indices.add(index)
        for key in ("source_atom_id", "source_residue_id"):
            _uuid(atom[key])
        _require(atom["source_atom_id"] not in atom_ids, "source_identity_mismatch")
        atom_ids.add(atom["source_atom_id"])
        _require(atom["element"] is None or isinstance(atom["element"], str) and 0 < len(atom["element"]) <= 256, "source_identity_mismatch")
        position = atom["position_nm"]
        _require(isinstance(position, list) and len(position) == 3
                 and all(type(v) in {int, float} and math.isfinite(v) for v in position), "source_identity_mismatch")


def write_terminal_result(request, output_dir, *, error=None, prepared=None,
                          resolved_parameters=None, qc=None, artifacts=None, error_message=None):
    """Write declared terminal outputs; success requires real P06 verification evidence."""
    artifacts = artifacts or {}
    if error is None:
        _require(isinstance(prepared, dict) and prepared.get("verification") == "passed", "result_invalid")
        _closed(prepared, {"revision_id", "manifest_ref", "verification", "eligibility"})
        _uuid(prepared["revision_id"])
        reference = prepared["manifest_ref"]
        _require(isinstance(reference, dict) and _valid_path(reference.get("member_path"))
                 and isinstance(reference.get("sha256"), str) and HASH.fullmatch(reference["sha256"])
                 and type(reference.get("size_bytes")) is int and reference["size_bytes"] >= 0, "result_invalid")
        eligibility = prepared["eligibility"]
        _closed(eligibility, {"eligible", "profile", "profile_revision", "reasons"})
        _require(type(eligibility["eligible"]) is bool and eligibility["profile"] == PROFILE
                 and type(eligibility["profile_revision"]) is int and eligibility["profile_revision"] == 1
                 and isinstance(eligibility["reasons"], list)
                 and all(isinstance(reason, str) and 0 < len(reason) <= 256 for reason in eligibility["reasons"]), "result_invalid")
    else:
        _require(prepared is None and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error) is not None, "result_invalid")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    descriptors = []
    with zipfile.ZipFile(output / "artifacts.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, item in sorted(artifacts.items()):
            _require(_valid_path(name), "result_invalid")
            data, media_type = item
            _require(isinstance(data, bytes) and isinstance(media_type, str), "result_invalid")
            archive.writestr(name, data)
            descriptors.append({"member_path": name, "sha256": sha256(data), "size_bytes": len(data), "media_type": media_type})
    provenance = {}
    for key in ("scientific_input_sha256", "native_config_sha256", "prepare_yaml_sha256", "source_manifest_sha256"):
        value = request.get(key)
        provenance[key] = value if isinstance(value, str) and HASH.fullmatch(value) else None
    provenance["config_contract"] = CONTRACT if request.get("config_contract") == CONTRACT else None
    provenance["execution_profile"] = PROFILE if request.get("execution_profile") == PROFILE else None
    provenance["profile_revision"] = 1 if type(request.get("profile_revision")) is int and request["profile_revision"] == 1 else None
    binding = request.get("executor_binding")
    revision = binding.get("git_revision") if isinstance(binding, dict) else None
    revision = revision if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) else None
    result = {"schema_version": "neomd.preparation-result/v1", "outcome": "scientific_failed" if error else "prepared",
              **provenance, "executor_git_revision": revision,
              "resolved_parameters": resolved_parameters, "qc": qc, "prepared": prepared,
              "error": {"code": error, "message": error_message or "Preparation did not complete.", "field_path": []} if error else None,
              "artifacts": descriptors}
    try:
        result, result_bytes = serialize_terminal_result(result)
    except BatchPreparationError:
        (output / "result.json").unlink(missing_ok=True)
        raise
    (output / "result.json").write_bytes(result_bytes)
    return result


def run_batch_preparation(request_path="/inputs/request.json", source_bundle="/inputs/source_bundle.zip",
                          work_dir="/work", output_dir="/outputs", *, executor=None,
                          diagnostic_observer=None):
    """Handle entered-wrapper failures; an absent scientific executor can never succeed."""
    request = {}
    stage = "request_read"

    def observe(error):
        if diagnostic_observer is not None:
            try:
                diagnostic_observer(stage, error)
            except Exception:
                pass  # Private diagnostics cannot change terminal behavior.

    try:
        with open(request_path, "rb") as handle:
            raw = handle.read(MAX_REQUEST + 1)
        if len(raw) <= MAX_REQUEST:
            candidate = _json(raw)
            if isinstance(candidate, dict):
                request = candidate
        stage = "materialize"
        materialized = materialize_preparation(request_path, source_bundle, work_dir)
        stage = "scientific_executor"
        if executor is None:
            raise BatchPreparationError("executor_unavailable")
        result = executor(materialized)
        stage = "terminal_result"
        write_terminal_result(request, output_dir, **result)
    except BatchPreparationError as error:
        observe(error)
        if error.code == "result_size_exceeded":
            # Transport capacity failure: retain archive evidence, publish no result.
            return 1
        write_terminal_result(request, output_dir, error=error.code)
    except Exception as error:
        observe(error)
        write_terminal_result(request if isinstance(request, dict) else {}, output_dir, error="wrapper_internal_error")
    return 0
