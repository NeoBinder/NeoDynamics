"""Versioned native preparation inspection; no source files or science are executed.

The ordered typed tree preserves native YAML loader semantics and omissions.
See contracts/native-preparation/v1 and docs/preparation-tools.md.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import rfc8785
import yaml
from yaml.events import CollectionEndEvent, CollectionStartEvent, ScalarEvent
from yaml.nodes import MappingNode, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, TagToken

CONTRACT = "neomd.native-prepare/v1"
PROFILE = "classic-source-box/v1"
MAX_BYTES = 32768
MAX_INTEGER = 9007199254740991
__all__ = ["CONTRACT", "PROFILE", "contract_directory", "inspect_native_preparation",
           "load_native_preparation", "export_native_json", "NativePreparationError"]


class NativePreparationError(ValueError):
    """Controlled diagnostics; the source document is retained by inspection."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(issue["code"] for issue in issues))


CONTRACT_DIR_ENV = "NEOMD_NATIVE_PREPARATION_CONTRACT_DIR"


def contract_directory() -> Path:
    """Locate the distribution's contract in an editable checkout or wheel.

    The NEOMD_NATIVE_PREPARATION_CONTRACT_DIR environment variable overrides
    both discovery paths for a pip install --target deployment whose data
    files do not sit where importlib.metadata expects them; an explicit
    override is validated rather than silently falling back.
    """
    override = os.environ.get(CONTRACT_DIR_ENV)
    if override:
        directory = Path(override)
        if not (directory / "native-fields.json").is_file():
            raise RuntimeError(
                f"{CONTRACT_DIR_ENV} does not contain native-fields.json: {directory}")
        return directory
    # This module is nested two packages deeper than the engine module it was
    # ported from (neomd/native_preparation.py), so the checkout root is
    # three parents up from this file.
    checkout = Path(__file__).resolve().parents[3] / "contracts/native-preparation/v1"
    if checkout.is_dir():
        return checkout
    distribution = importlib.metadata.distribution("NeoDynamics")
    for file in distribution.files or ():
        if str(file).endswith("share/neodynamics/contracts/native-preparation/v1/native-fields.json"):
            return Path(distribution.locate_file(file)).parent
    raise RuntimeError("native preparation contract is absent from the installation")


def _issue(code, path=(), message="Native configuration does not satisfy the contract."):
    return {"code": code, "field_path": list(path), "message": message}


def _compact(tree):
    return json.dumps(tree, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _parse(text):
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise NativePreparationError([_issue("request_size_exceeded")])
    loader = yaml.SafeLoader(text)
    count = 0

    def walk(node, path=(), depth=0):
        nonlocal count
        count += 1
        if depth > 32 or count > 10000:
            raise NativePreparationError([_issue("native_invalid_yaml", path, "Native input exceeds parsing limits.")])
        if isinstance(node, ScalarNode):
            tag = node.tag.removeprefix("tag:yaml.org,2002:")
            if tag not in {"null", "bool", "int", "float", "str"}:
                raise NativePreparationError([_issue("native_invalid_yaml", path)])
            value = loader.construct_object(node)
            if tag == "str":
                try:
                    value.encode("utf-8")
                except UnicodeError:
                    raise NativePreparationError([_issue(
                        "native_invalid_yaml", path, "Native strings must be valid UTF8.")]) from None
            if tag == "int" and abs(value) > MAX_INTEGER:
                raise NativePreparationError([_issue("native_type_mismatch", path, "Integer exceeds the exact representation limit.")])
            if tag == "float" and not math.isfinite(value):
                raise NativePreparationError([_issue("native_type_mismatch", path, "Numbers must be finite.")])
            encoded = str(value) if tag == "int" else (
                rfc8785.dumps(value).decode() if tag == "float" else value)
            return value, ["string" if tag == "str" else tag, encoded]
        if isinstance(node, SequenceNode):
            pairs = [walk(child, (*path, i), depth + 1) for i, child in enumerate(node.value)]
            return [pair[0] for pair in pairs], ["seq", [pair[1] for pair in pairs]]
        if isinstance(node, MappingNode):
            data, entries, seen = {}, [], set()
            for key, child in node.value:
                if not isinstance(key, ScalarNode):
                    raise NativePreparationError([_issue("native_invalid_yaml", path)])
                key_value, key_tree = walk(key, path, depth + 1)
                integer_location = (len(path) == 3 and path[:2] == ("protein", "custom_resname_dict"))
                if key_tree[0] != "string" and not (key_tree[0] == "int" and integer_location):
                    raise NativePreparationError([_issue("native_type_mismatch", path, "Mapping key type is not declared at this field.")])
                fingerprint = _compact(key_tree)
                if fingerprint in seen:
                    raise NativePreparationError([_issue("native_duplicate_key", (*path, key_value))])
                seen.add(fingerprint)
                value, tree = walk(child, (*path, key_value), depth + 1)
                data[key_value] = value
                entries.append([key_tree, tree])
            return data, ["map", entries]
        raise NativePreparationError([_issue("native_invalid_yaml", path)])

    try:
        if any(isinstance(token, (AliasToken, AnchorToken, TagToken)) for token in yaml.scan(text)):
            raise NativePreparationError([_issue("native_invalid_yaml", message="Aliases, anchors and explicit tags are unsupported.")])
        depth, nodes = -1, 0
        for event in yaml.parse(text):
            if isinstance(event, CollectionStartEvent):
                depth += 1
                nodes += 1
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
            elif isinstance(event, ScalarEvent):
                nodes += 1
            if depth > 32 or nodes > 10000:
                raise NativePreparationError([_issue("native_invalid_yaml", message="Native input exceeds parsing limits.")])
        node = loader.get_single_node()
        if node is None:
            return None, ["null", None]
        return walk(node)
    except (yaml.YAMLError, ValueError, OverflowError) as error:
        if isinstance(error, NativePreparationError):
            raise
        raise NativePreparationError([_issue("native_invalid_yaml")]) from None
    finally:
        loader.dispose()


def _matches(kind, value):
    if kind in {"string", "nonempty string"}:
        return isinstance(value, str) and (kind == "string" or bool(value.strip()))
    if kind == "boolean":
        return type(value) is bool
    if kind.startswith("finite number"):
        return (type(value) in {int, float} and math.isfinite(value)
                and (value >= 0 if ">=0" in kind else value > 0))
    if kind.startswith("enum["):
        return isinstance(value, str) and value in kind[5:-1].split(",")
    if kind == "list[string]":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if kind == "list[map[integer>=1,string]]":
        return isinstance(value, list) and all(isinstance(item, dict) and all(
            type(key) is int and key >= 1 and isinstance(val, str)
            for key, val in item.items()) for item in value)
    if kind == "list[list[string,string]]":
        return isinstance(value, list) and all(isinstance(item, list) and len(item) == 2
                                             and all(isinstance(val, str) for val in item)
                                             for item in value)
    raise RuntimeError("unimplemented distributed field type")


def _validate(data, profile):
    vocabulary = json.loads((contract_directory() / "native-fields.json").read_text())
    issues = []
    fields = [(field["path"].split("."), field) for field in vocabulary["fields"]]
    unsupported = set(vocabulary["unsupported_consumed_metadata"] + vocabulary["unsupported_ignored_keys"])

    def visit(value, path=()):
        candidates = [(parts, field) for parts, field in fields if len(parts) >= len(path)
                      and all(part == key or part.startswith("<")
                              for part, key in zip(parts, path))]
        exact = [field for parts, field in candidates if len(parts) == len(path)]
        if exact:
            field = exact[0]
            role = field["role"]
            path_valid = True
            if role in {"input_file", "input_directory", "output_directory", "builtin_resource"}:
                path_valid = isinstance(value, str) and bool(value.strip())
            elif role == "input_file_list":
                path_valid = isinstance(value, list) and all(
                    isinstance(item, str) and bool(item.strip()) for item in value)
            if not _matches(field["type"], value) or not path_valid:
                issues.append(_issue("native_type_mismatch", path))
            if profile and not field["online"]:
                issues.append(_issue("native_unsupported_field", path))
            if profile and field["role"] == "builtin_resource" and value != field["default"]:
                issues.append(_issue("native_unsupported_field", path))
            return
        if not candidates:
            code = "native_unsupported_field" if ".".join(path) in unsupported else "native_unknown_field"
            issues.append(_issue(code, path))
            return
        if profile and all(not field["online"] for _, field in candidates):
            issues.append(_issue("native_unsupported_field", path))
        if not isinstance(value, dict):
            issues.append(_issue("native_type_mismatch", path))
            return
        for key, child in value.items():
            visit(child, (*path, key))

    visit(data)
    if not isinstance(data, dict):
        return issues
    branches = [name for name in ("from_gromacs", "from_amber") if name in data]
    classic = "protein" in data or "ligands" in data
    if len(branches) + int(classic) != 1:
        issues.append(_issue("native_type_mismatch", message="Exactly one preparation branch is required."))
    required = [("output_dir",)]
    if "protein" in data or profile:
        required.append(("protein", "path"))
    for branch in branches:
        if profile:
            issues.append(_issue("native_unsupported_field", (branch,)))
        required.extend(tuple(field["path"].split(".")) for field in vocabulary["fields"]
                        if field["path"].startswith(branch + "."))
    ligands = data.get("ligands")
    if isinstance(ligands, dict):
        if "protein" not in data and not ligands:
            issues.append(_issue("native_type_mismatch", ("ligands",), "A classic source is required."))
        for key in ligands:
            if not key.strip():
                issues.append(_issue("native_type_mismatch", ("ligands", key)))
            required.append(("ligands", key, "path"))
            if profile:
                required.extend(("ligands", key, name) for name in ("smiles", "resname"))
    for path in required:
        value = data
        for key in path:
            if not isinstance(value, dict) or key not in value:
                issues.append(_issue("native_type_mismatch", path, "Required native field is absent."))
                break
            value = value[key]
    return issues


@dataclass(frozen=True)
class NativePreparationInspection:
    original_text: str
    data: object
    typed_tree: object
    native_typed_sha256: str | None
    yaml_sha256: str | None
    issues: list

    @property
    def valid(self):
        return not self.issues

    def as_dict(self):
        return {"config_contract": CONTRACT, "valid": self.valid,
                "native_typed_sha256": self.native_typed_sha256, "yaml_sha256": self.yaml_sha256,
                "typed_tree": self.typed_tree, "issues": self.issues,
                "scientific_qualification": "not_checked"}


def inspect_native_preparation(text: str, *, execution_profile: str | None = None):
    """Inspect exact UTF8 YAML text without reading paths, filling defaults or running science."""
    try:
        yaml_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeError:
        return NativePreparationInspection(text, None, None, None, None,
                                           [_issue("native_invalid_yaml", message="Native input must be valid UTF8.")])
    try:
        data, tree = _parse(text)
    except NativePreparationError as error:
        return NativePreparationInspection(text, None, None, None, yaml_hash, error.issues)
    issues = _validate(data, execution_profile)
    if execution_profile not in {None, PROFILE}:
        issues.append(_issue("profile_unsupported"))
    return NativePreparationInspection(text, data, tree, hashlib.sha256(_compact(tree)).hexdigest(),
                                       yaml_hash, issues)


def load_native_preparation(text: str, *, execution_profile: str | None = None) -> dict:
    """Return checked native values; raise controlled issues without discarding the draft."""
    inspection = inspect_native_preparation(text, execution_profile=execution_profile)
    if not inspection.valid:
        raise NativePreparationError(inspection.issues)
    return inspection.data


def export_native_json(text: str) -> str:
    """Export only when JSON survives the actual YAML loader with an identical typed tree."""
    inspection = inspect_native_preparation(text)
    if not inspection.valid:
        raise NativePreparationError(inspection.issues)
    candidate = json.dumps(inspection.data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        _, tree = _parse(candidate)
    except NativePreparationError:
        tree = None
    if tree != inspection.typed_tree:
        raise NativePreparationError([_issue("json_native_type_mismatch", message="This configuration requires YAML; retain the original draft.")])
    return candidate
