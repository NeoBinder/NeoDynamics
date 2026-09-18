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
    for folder in ("scientific-protocol/v1", "prepared-bundle/v1", "prepared-bundle/v2",
                   "protein-preparation/v1", "trajectory-result/v1", "topology-view/v1"):
        for path in (contracts / folder).glob("*.schema.json"):
            scientific = json.loads(path.read_text())
            scientific_schemas[scientific["$id"]] = scientific
            registry = registry.with_resource(scientific["$id"], Resource.from_contents(scientific))
    if name in scientific_schemas:
        selected = scientific_schemas[name]
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
    if problems:
        raise ContractError(problems)
    return value
