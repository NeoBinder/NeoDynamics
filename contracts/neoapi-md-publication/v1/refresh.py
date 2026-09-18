"""Refresh only this attributed consumer snapshot from an explicit NeoAPI OpenAPI root."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

SOURCE_SHA256 = "7c43f5341e6effa21d8c8731418ceeb9ac87ba1d21cf3e1a14b250c21335ea22"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def rewrite(value):
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "$ref":
                if not item.startswith("#/components/schemas/"):
                    raise ValueError("publication schema requires a nonlocal reference")
                result[key] = item.replace("#/components/schemas/", "#/$defs/", 1)
            else:
                result[key] = rewrite(item)
        return result
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openapi_root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    original = (args.openapi_root / "internal/md-publication-v1.yaml").read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise SystemExit("upstream publication contract differs from the reviewed version")
    source = yaml.safe_load(original)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:neodynamics:consumer:neoapi-md-publication:v1",
        "$defs": rewrite(source["components"]["schemas"]),
    }
    files = {"openapi.yaml": original, "schema.json": encoded(schema)}
    for name in ("sealed-output-set", "different-generation", "different-preparation"):
        for suffix in (".json", ".jcs", ".sha256"):
            relative = "fixtures/" + name + suffix
            files[relative] = (args.openapi_root / "fixtures/md-publication-v1" / (name + suffix)).read_bytes()
    inventory = {
        "schema_version": 1,
        "source_repository": "NeoAPI-rs",
        "source_path": "openapi/internal/md-publication-v1.yaml",
        "source_sha256": SOURCE_SHA256,
        "derivation": "components.schemas copied; local refs rewritten to $defs; no semantic changes",
        "error_statuses": source["x-error-statuses"],
        "files": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }
    files["snapshot.json"] = encoded(inventory)
    for name, body in files.items():
        path = destination / name
        if args.check:
            if not path.is_file() or path.read_bytes() != body:
                raise SystemExit("publication consumer snapshot differs: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)


if __name__ == "__main__":
    main()
