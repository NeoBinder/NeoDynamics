"""Prepared schema validation with bundled, finite references resolved once."""

from functools import lru_cache

import jsonschema

from .contract import ContractError, canonical_bytes, validator

_KINDS = ("manifest", "mapping")
_VERSIONS = (1, 2)
_SINGLE = {"additionalProperties", "unevaluatedProperties", "propertyNames", "items",
           "contains", "unevaluatedItems", "not", "if", "then", "else", "contentSchema"}
_ARRAY = {"allOf", "anyOf", "oneOf", "prefixItems"}
_MAP = {"properties", "patternProperties", "dependentSchemas", "$defs"}


@lru_cache(maxsize=1)
def _validators():
    documents = {
        f"urn:neodyn:prepared-bundle:v{version}:{kind}":
        validator(f"urn:neodyn:prepared-bundle:v{version}:{kind}").schema
        for version in _VERSIONS for kind in _KINDS
    }
    resolved = {}
    visiting = set()

    def reference(text, document):
        uri, separator, fragment = text.partition("#")
        uri = uri or document
        # These shipped schemas have only direct $defs pointers, no external
        # retrieval, nested resource identities, anchors or recursive references.
        if uri not in documents or not separator or not fragment.startswith("/$defs/"):
            raise RuntimeError("prepared schema reference requires qualification")
        key = uri, fragment
        if key in visiting:
            raise RuntimeError("recursive prepared schema requires qualification")
        if key not in resolved:
            visiting.add(key)
            target = documents[uri]
            for token in fragment[1:].split("/"):
                target = target[token.replace("~1", "/").replace("~0", "~")]
            resolved[key] = expand(target, uri)
            visiting.remove(key)
        return resolved[key]

    def expand(schema, document, *, root=False):
        if isinstance(schema, bool):
            return schema
        if (not root and "$id" in schema) or any(
            key in schema for key in ("$anchor", "$dynamicAnchor", "$dynamicRef",
                                     "$recursiveAnchor", "$recursiveRef")
        ):
            raise RuntimeError("prepared schema scope requires qualification")
        if "$ref" in schema:
            # Do not silently lose Draft 2020-12 sibling/annotation semantics if
            # a later schema introduces a pattern this finite expansion lacks.
            if set(schema) != {"$ref"}:
                raise RuntimeError("prepared schema reference siblings require qualification")
            return reference(schema["$ref"], document)
        output = {}
        for key, value in schema.items():
            if key in _SINGLE:
                output[key] = expand(value, document)
            elif key in _ARRAY:
                output[key] = [expand(item, document) for item in value]
            elif key in _MAP:
                output[key] = {name: expand(item, document) for name, item in value.items()}
            else:
                output[key] = value
        return output

    return {
        name: jsonschema.Draft202012Validator(
            expand(schema, name, root=True), format_checker=jsonschema.FormatChecker()
        )
        for name, schema in documents.items()
    }


def validate_prepared(kind, value):
    """Validate manifest/mapping v1 or v2 and JCS, returning the unchanged input.

    Errors retain collect-all instance paths. This checks document schema only;
    part hashes, scientific correspondence and authenticity remain caller checks.
    Only the four shipped schemas are accepted, never caller-supplied schemas.
    """
    if kind not in _KINDS:
        raise ValueError("unknown prepared schema kind")
    version = value.get("schema_version") if isinstance(value, dict) else None
    problems = []
    if version not in _VERSIONS:
        problems.append(("/schema_version", "prepared schema version must be 1 or 2"))
    else:
        # Numeric 1.0/2.0 have the same JSON Schema const meaning as 1/2. A boolean
        # must still pass the schema itself: Python's True == 1 is not validation.
        name = f"urn:neodyn:prepared-bundle:v{int(version)}:{kind}"
        problems.extend(
            ("/" + "/".join(map(str, error.absolute_path)), error.message)
            for error in _validators()[name].iter_errors(value)
        )
    try:
        canonical_bytes(value)
    except (ValueError, TypeError) as error:
        problems.append(("/", str(error)))
    if problems:
        raise ContractError(problems)
    return value
