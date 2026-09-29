"""Versioned transport validation and RFC 8785 request hashing."""

from __future__ import annotations

import hashlib
import json
import sysconfig
from functools import lru_cache
from pathlib import Path

import jsonschema
import rfc8785
from referencing import Registry, Resource


class ContractError(ValueError):
    """Public-safe collect-all protocol problems."""

    def __init__(self, problems):
        self.problems = tuple(problems)
        super().__init__("; ".join(f"{path}: {message}" for path, message in self.problems))


def canonical_bytes(value):
    """JCS bytes; invalid Unicode, nonfinite values and unsafe integers fail."""
    return rfc8785.dumps(value)


def content_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@lru_cache(maxsize=32)
def validator(name):
    contracts = Path(__file__).resolve().parents[2] / "contracts"
    if not contracts.exists():
        contracts = Path(sysconfig.get_path("data")) / "share/neodynamics/contracts"
    schema = json.loads((contracts / "imd/v1/protocol.schema.json").read_text())
    registry = Registry()
    scientific_schemas = {}
    for folder in ("scientific-protocol/v1", "scientific-protocol-template/v1",
                   "prepared-bundle/v1", "prepared-bundle/v2",
                   "protein-preparation/v1", "protein-preparation/v2",
                   "component-operations/v1",
                   "molecular-topology/v1", "trajectory-result/v1", "topology-view/v1",
                   "trajectory-comparison/v1"):
        for path in (contracts / folder).glob("*.schema.json"):
            scientific = json.loads(path.read_text())
            scientific_schemas[scientific["$id"]] = scientific
            registry = registry.with_resource(scientific["$id"], Resource.from_contents(scientific))
    if name in scientific_schemas:
        selected = scientific_schemas[name]
    elif "#/$defs/" in name:
        base, _, definition = name.partition("#/$defs/")
        definitions = scientific_schemas.get(base, {}).get("$defs", {})
        if definition not in definitions:
            raise ValueError("unknown protocol schema")
        selected = {"$ref": f"#/$defs/{definition}", "$defs": definitions}
    elif name in schema["$defs"]:
        selected = {"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]}
    else:
        raise ValueError("unknown protocol schema")
    return jsonschema.Draft202012Validator(
        selected,
        format_checker=jsonschema.FormatChecker(),
        registry=registry,
    )


def validate(name, value):
    problems = [
        ("/" + "/".join(map(str, error.absolute_path)), error.message)
        for error in validator(name).iter_errors(value)
    ]
    try:
        canonical_bytes(value)
    except (ValueError, TypeError) as error:
        problems.append(("/", str(error)))
    if name in {"CommandRequest", "EngineCommand"} and not problems:
        payload = value.get("payload", {}) if name == "CommandRequest" else value
        ids = payload.get("atom_ids", [])
        vectors = payload.get("vectors", payload.get("targets", []))
        if len(ids) != len(vectors):
            problems.append(("/payload/atom_ids", "one xyz vector is required per AtomId"))
        if payload.get("type") == "set_forces":
            try:
                if any(sum(float(x) ** 2 for x in row) > 10000.0**2 for row in vectors):
                    problems.append(("/payload/vectors", "force norm exceeds 10000 kJ/mol/nm"))
            except (TypeError, ValueError, OverflowError):
                problems.append(("/payload/vectors", "invalid force vector"))
    if name in {"CreateRequest", "CommandRequest"} and not problems:
        material = (
            {key: value[key] for key in ("session_id", "payload")}
            if name == "CreateRequest"
            else {key: value[key] for key in ("expected", "controller", "payload")}
        )
        if value["payload_sha256"] != content_hash(material):
            problems.append(("/payload_sha256", "content does not match canonical payload hash"))
    if name == "InputOperation" and not problems and value["state"] == "succeeded":
        result = value["result"]
        if result["prepared"] != value["prepared"]:
            problems.append(("/result/prepared", "result differs from operation prepared binding"))
        if value["kind"] == "prepared_commit" and result["prepared"]["bundle_id"] != value["upload_id"]:
            problems.append(("/result/prepared/bundle_id", "result differs from the committed upload"))
        if value["kind"] == "protocol_register" and (
            result["protocol_id"] != value["protocol_id"] or result["request_id"] != value["request_id"]
        ):
            problems.append(("/result", "protocol result differs from operation identity"))
    if name in {
        "urn:neodyn:component-operations:v1:operation#/$defs/ComponentRequest",
        "urn:neodyn:component-operations:v1:operation",
    } and not problems:
        request = value if name.endswith("ComponentRequest") else value["request"]
        seen = set()
        for index, binding in enumerate(request["inputs"]):
            logical_name = binding["name"]
            if logical_name in seen:
                prefix = "/request" if name.endswith(":operation") else ""
                problems.append((f"{prefix}/inputs/{index}/name", "duplicate logical input name"))
            if logical_name == "index.json" or logical_name.startswith("index.json/"):
                prefix = "/request" if name.endswith(":operation") else ""
                problems.append((f"{prefix}/inputs/{index}/name", "reserved logical input name"))
            if any(logical_name.startswith(previous + "/") or
                   previous.startswith(logical_name + "/") for previous in seen):
                prefix = "/request" if name.endswith(":operation") else ""
                problems.append((f"{prefix}/inputs/{index}/name", "logical input name collides with a directory"))
            seen.add(logical_name)
        if name.endswith(":operation") and (
            value["request_id"] != request["request_id"] or value["kind"] != request["kind"]
        ):
            problems.append(("/request", "request identity differs from operation identity"))
        if name.endswith(":operation"):
            if value["request_sha256"] != content_hash(request):
                problems.append(("/request_sha256", "request hash differs from canonical request"))
            receipt = value["receipt"]
            if receipt is not None and any(
                receipt[key] != value[key] for key in ("operation_id", "request_id", "kind")
            ):
                problems.append(("/receipt", "receipt identity differs from operation identity"))
            if receipt is not None:
                bindings = receipt["input_bindings"]
                if len(bindings) != len(request["inputs"]):
                    problems.append(("/receipt/input_bindings", "input count differs from request"))
                else:
                    for index, (source, binding) in enumerate(zip(request["inputs"], bindings)):
                        identity = ("name", "sha256", "size_bytes")
                        references = ("upload_id",) if "upload_id" in source else ("operation_id", "artifact_id")
                        if (any(source[key] != binding[key] for key in identity) or
                                any(source[key] != binding["source"].get(key) for key in references) or
                                binding["resolved_owner_scope"] != receipt["owner_scope"]):
                            problems.append((f"/receipt/input_bindings/{index}",
                                             "resolved input differs from request or owner"))
    if problems:
        raise ContractError(problems)
    return value
