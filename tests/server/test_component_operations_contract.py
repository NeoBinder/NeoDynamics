"""Private component operation transport is closed, bounded, and lineage-safe."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from neomd_server.contract import ContractError, content_hash, validate

ROOT = Path(__file__).resolve().parents[2] / "contracts/component-operations/v1"
OPERATION = "urn:neodyn:component-operations:v1:operation"
REQUEST = OPERATION + "#/$defs/ComponentRequest"
BEGIN = OPERATION + "#/$defs/BlobUploadBeginRequest"
UPLOAD = OPERATION + "#/$defs/BlobUploadReceipt"
UUID = "00000000-0000-4000-8000-000000000001"
OTHER = "00000000-0000-4000-8000-000000000002"
SHA = "a" * 64
NOW = "2026-09-28T00:00:00Z"


def fixture(name):
    return json.loads((ROOT / "fixtures" / name).read_text())


def test_schema_documents_are_valid():
    for path in ROOT.glob("*.schema.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_available_catalog_rows_require_bounded_scope():
    catalog_id = "urn:neodyn:component-operations:v1:catalog"
    row = {"kind": "component_import", "method_id": "declarative",
           "availability": "available", "qualification_sha256": SHA,
           "scope": {"chemistry_scope": "neutral JZ4 fixture", "states": [
               {"state_sha256": SHA, "formal_charge": 0}], "platform": "CPU",
               "resource_sha256": SHA, "receipt_sha256": SHA}}
    catalog = {"schema_version": 1, "methods": [row], "limits": {
        "max_input_count": 64, "max_blob_bytes": 67108864,
        "max_result_bytes": 1000000}}
    validate(catalog_id, catalog)
    for field in ("scope", "qualification_sha256"):
        changed = deepcopy(catalog)
        del changed["methods"][0][field]
        with pytest.raises(ContractError):
            validate(catalog_id, changed)
    changed = deepcopy(catalog)
    changed["methods"][0]["scope"]["platform"] = "CUDA"
    with pytest.raises(ContractError):
        validate(catalog_id, changed)
    changed = deepcopy(catalog)
    changed["methods"][0] = {"kind": "component_import",
                             "method_id": "declarative", "availability": "unavailable"}
    validate(catalog_id, changed)
    changed["methods"][0]["scope"] = row["scope"]
    with pytest.raises(ContractError):
        validate(catalog_id, changed)
    changed = deepcopy(catalog)
    changed["methods"][0].update(kind="component_assemble", method_id="stored-package-v1")
    with pytest.raises(ContractError):
        validate(catalog_id, changed)
    changed["methods"][0]["scope"]["profiles"] = [
        {"source_sha256": SHA, "state_sha256s": [SHA, SHA],
         "package_methods": [{"method_id": "amber.gaff2.abcg2", "resource_sha256": SHA}],
         "resource_sha256": SHA, "receipt_sha256": SHA}]
    validate(catalog_id, changed)


def test_component_errors_have_closed_http_statuses():
    from neomd_server.app import _error_statuses

    statuses = _error_statuses()
    for code in ("contract_version_mismatch", "invalid_reference", "oversized_payload",
                 "mapping_mismatch", "chemistry_unresolved", "incompatible_package",
                 "capability_unavailable", "stereochemistry_unresolved"):
        assert statuses[code] in {409, 413, 422, 503}


@pytest.mark.parametrize("name", [
    "component_review.request.json", "component_generate.request.json",
    "component_generate_protein.request.json", "component_import.request.json",
    "component_generate_sep.request.json",
    "component_assemble.request.json",
])
def test_fixed_request_fixtures(name):
    validate(REQUEST, fixture(name))


@pytest.mark.parametrize("name", [
    "invalid-traversal.request.json", "invalid-duplicate.request.json",
    "invalid-reserved.request.json",
])
def test_unsafe_request_fixtures(name):
    with pytest.raises(ContractError):
        validate(REQUEST, fixture(name))


def test_input_refs_and_options_reject_ambiguity():
    request = fixture("component_generate.request.json")
    bad = deepcopy(request)
    bad["inputs"][0]["operation_id"] = OTHER
    with pytest.raises(ContractError):
        validate(REQUEST, bad)
    bad = deepcopy(request)
    bad["inputs"][0]["name"] = "states/.secret.json"
    with pytest.raises(ContractError):
        validate(REQUEST, bad)
    bad = deepcopy(request)
    bad["options"]["extra"] = "unbounded"
    with pytest.raises(ContractError):
        validate(REQUEST, bad)
    bad = deepcopy(request)
    bad["inputs"].append({**bad["inputs"][0], "name": "states"})
    with pytest.raises(ContractError):
        validate(REQUEST, bad)
    bad = deepcopy(request)
    bad["inputs"][0]["name"] = "index.json/child"
    with pytest.raises(ContractError):
        validate(REQUEST, bad)


def test_sage_generate_uses_the_same_closed_seeded_request_shape():
    request = fixture("component_generate.request.json")
    request["options"]["method_id"] = "openff-sage-2.2.1"
    validate(REQUEST, request)
    request["options"]["method_id"] = "unqualified-fallback"
    with pytest.raises(ContractError):
        validate(REQUEST, request)


def test_sep_generate_has_one_closed_resource_and_residue_selection():
    request = fixture("component_generate_sep.request.json")
    validate(REQUEST, request)
    for field, value in (("template_id", "SER"),
                         ("template_resource_sha256", "0" * 64),
                         ("source_residue_id", "0")):
        changed = deepcopy(request)
        changed["options"][field] = value
        with pytest.raises(ContractError):
            validate(REQUEST, changed)
    changed = deepcopy(request)
    changed["options"]["repair_seed"] = 123
    with pytest.raises(ContractError):
        validate(REQUEST, changed)


def test_blob_receipt_and_operation_identity():
    begin = fixture("blob-upload.begin.json")
    validate(BEGIN, begin)
    upload = {**begin, "upload_id": OTHER, "state": "stored", "created_at": NOW}
    validate(UPLOAD, upload)
    request = fixture("component_import.request.json")
    operation = {"schema_version": 1, "operation_id": OTHER,
                 "request_id": request["request_id"], "kind": request["kind"],
                 "revision": 1, "state": "queued", "request": request,
                 "request_sha256": content_hash(request), "receipt": None,
                 "error": None, "created_at": NOW, "updated_at": NOW}
    validate(OPERATION, operation)
    wrong = {**operation, "request_id": UUID}
    with pytest.raises(ContractError):
        validate(OPERATION, wrong)
    wrong = {**operation, "request_sha256": SHA}
    with pytest.raises(ContractError):
        validate(OPERATION, wrong)
    wrong = {**operation, "state": "succeeded"}
    with pytest.raises(ContractError):
        validate(OPERATION, wrong)


def test_success_receipt_inventory_and_qc():
    request = fixture("component_import.request.json")
    artifact = {"artifact_id": UUID, "role": "package", "sha256": SHA, "size_bytes": 300}
    receipt = {
        "schema_version": 1, "receipt_id": OTHER, "operation_id": UUID,
        "request_id": request["request_id"], "kind": request["kind"],
        "owner_scope": UUID, "backend": {"provider": "neodynamics", "pool_id": "cpu"},
        "input_bindings": [{"name": "states/ligand.json", "sha256": SHA,
                            "size_bytes": 300, "resolved_owner_scope": UUID,
                            "source": {"upload_id": request["inputs"][0]["upload_id"]}}],
        "result": {"object_type": "parameter_package", "id": UUID, "sha256": SHA},
        "verification": {"profile": "component_import_v1",
                         "verifier": "neomd_server.component_preparation_execution",
                         "report_sha256": SHA, "verified_at": NOW},
        "artifacts": [artifact], "created_at": NOW,
    }
    validate("urn:neodyn:component-operations:v1:receipt", receipt)
    operation = {"schema_version": 1, "operation_id": UUID,
                 "request_id": request["request_id"], "kind": request["kind"],
                 "revision": 3, "state": "succeeded", "request": request,
                 "request_sha256": content_hash(request), "receipt": receipt,
                 "error": None, "created_at": NOW, "updated_at": NOW}
    validate(OPERATION, operation)
    mismatched = deepcopy(operation)
    mismatched["receipt"]["input_bindings"][0]["name"] = "other.json"
    with pytest.raises(ContractError):
        validate(OPERATION, mismatched)
    prior = deepcopy(receipt)
    prior["input_bindings"][0]["source"] = {"operation_id": UUID,
        "artifact_id": OTHER, "producer_receipt_sha256": SHA}
    validate("urn:neodyn:component-operations:v1:receipt", prior)
    inventory = {"schema_version": 1, "operation_id": UUID, "receipt_id": OTHER,
                 "artifacts": [{**artifact, "path": "package.json"}],
                 "total_size_bytes": 300}
    validate("urn:neodyn:component-operations:v1:inventory", inventory)
    qc = {"schema_version": 1, "operation_id": UUID, "receipt_id": OTHER,
          "outcome": "passed", "report_sha256": SHA, "summary": "verified",
          "blockers": []}
    validate("urn:neodyn:component-operations:v1:qc", qc)
    with pytest.raises(ContractError):
        validate("urn:neodyn:component-operations:v1:inventory",
                 {**inventory, "artifacts": [{**artifact, "path": "../secret"}]})


def test_component_http_routes_are_scoped_and_validated():
    import asyncio
    from types import SimpleNamespace

    import httpx

    from neomd_server.app import PREFIX, TransportSettings, create_app

    request = fixture("component_import.request.json")
    begin = fixture("blob-upload.begin.json")
    upload = {**begin, "upload_id": OTHER, "state": "registered", "created_at": NOW}
    operation = {"schema_version": 1, "operation_id": OTHER,
                 "request_id": request["request_id"], "kind": request["kind"],
                 "revision": 1, "state": "queued", "request": request,
                 "request_sha256": content_hash(request), "receipt": None,
                 "error": None, "created_at": NOW, "updated_at": NOW}
    catalog = {"schema_version": 1, "methods": [
        {"kind": "component_import", "method_id": "declarative", "availability": "available",
         "qualification_sha256": SHA,
         "scope": {"chemistry_scope": "neutral JZ4 fixture", "states": [
             {"state_sha256": SHA, "formal_charge": 0}], "platform": "CPU",
             "resource_sha256": SHA, "receipt_sha256": SHA}}],
        "limits": {"max_input_count": 64, "max_blob_bytes": 67108864,
                   "max_result_bytes": 1000000}}
    inventory = {"schema_version": 1, "operation_id": OTHER, "receipt_id": UUID,
                 "artifacts": [{"artifact_id": UUID, "role": "package", "path": "package.json",
                                "sha256": SHA, "size_bytes": 3}], "total_size_bytes": 3}
    qc = {"schema_version": 1, "operation_id": OTHER, "receipt_id": UUID,
          "outcome": "passed", "report_sha256": SHA, "summary": "verified", "blockers": []}

    class Ports:
        limits = SimpleNamespace(max_source_bytes=67108864)

        def __init__(self):
            self.calls = []

        async def verify(self, token):
            return token == "x" * 32

        async def start(self):
            pass

        async def initialize(self):
            pass

        async def close(self):
            pass

        async def component_catalog(self, scope):
            self.calls.append(("catalog", scope))
            return catalog

        async def begin_component_blob(self, scope, body):
            self.calls.append(("begin", scope, body))
            return upload

        async def component_blob_upload(self, scope, upload_id):
            self.calls.append(("upload", scope, upload_id))
            return upload

        async def component_blob_by_request(self, scope, request_id):
            self.calls.append(("upload_by_request", scope, request_id))
            return upload

        async def put_source_content(self, scope, upload_id, chunks):
            content = b"".join([chunk async for chunk in chunks])
            self.calls.append(("put", scope, upload_id, content))

        async def submit_component(self, scope, body):
            self.calls.append(("submit", scope, body))
            return operation

        async def component_by_request(self, scope, request_id):
            self.calls.append(("by_request", scope, request_id))
            return operation

        async def get_component(self, scope, operation_id):
            self.calls.append(("get", scope, operation_id))
            return operation

        async def cancel_component(self, scope, operation_id):
            self.calls.append(("cancel", scope, operation_id))
            return operation

        async def component_inventory(self, scope, operation_id):
            self.calls.append(("inventory", scope, operation_id))
            return inventory

        async def component_qc(self, scope, operation_id):
            self.calls.append(("qc", scope, operation_id))
            return qc

        async def artifact_content(self, scope, operation_id, artifact_id):
            self.calls.append(("artifact", scope, operation_id, artifact_id))
            return b"abc"

    async def scenario():
        ports = Ports()
        app = create_app(ports, ports, ports, TransportSettings(
            ports.verify, max_upload_part_bytes=1024, max_coordinate_upload_bytes=1024),
            preparations=ports)
        headers = {"Authorization": "Bearer " + "x" * 32, "X-Neomd-Scope": UUID}
        prefix = PREFIX + "/component-preparations"
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://test", headers=headers) as client:
                assert (await client.get(prefix + "/catalog")).json() == catalog
                assert (await client.post(prefix + "/blob-uploads", json=begin)).json() == upload
                assert (await client.get(prefix + "/blob-uploads/by-request/" + begin["request_id"])).json() == upload
                assert (await client.get(prefix + "/blob-uploads/" + OTHER)).json() == upload
                put = await client.put(prefix + "/blob-uploads/" + OTHER + "/content",
                                       content=b"abc", headers={"Content-Type": "application/octet-stream"})
                assert put.status_code == 200
                assert (await client.post(prefix, json=request)).status_code == 202
                assert (await client.get(prefix + "/by-request/" + request["request_id"])).json() == operation
                assert (await client.get(prefix + "/" + OTHER)).json() == operation
                assert (await client.post(prefix + "/" + OTHER + "/cancel")).json() == operation
                assert (await client.get(prefix + "/" + OTHER + "/inventory")).json() == inventory
                assert (await client.get(prefix + "/" + OTHER + "/qc")).json() == qc
                content = await client.get(prefix + "/" + OTHER + "/artifacts/" + UUID + "/content")
                assert content.content == b"abc"
                assert content.headers["cache-control"] == "no-store"
                invalid = deepcopy(request)
                invalid["inputs"][0]["name"] = "index.json"
                assert (await client.post(prefix, json=invalid)).status_code == 400
        assert ("put", UUID, OTHER, b"abc") in ports.calls
        assert ("artifact", UUID, OTHER, UUID) in ports.calls

    asyncio.run(scenario())
