"""Deletion wire identity, exact group selection and truthful lifecycle receipts."""

import json
from pathlib import Path

import pytest

from neomd_server.contract import ContractError, content_hash, validate

FIXTURES = Path(__file__).parents[2] / "contracts/imd/v1/fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("case", fixture("deletion-contract-cases.json"))
def test_frozen_deletion_documents_and_canonical_identity(case):
    value = fixture(case["file"])
    validate(case["schema"], value)
    assert content_hash(value) == case["canonical_sha256"]


@pytest.mark.parametrize("schema,file", [
    ("DeletionPreviewRequest", "deletion-accept.json"),
    ("AcceptSessionDeletion", "deletion-accept.json"),
    ("ServiceDeletionPreview", "deletion-preview.json"),
])
def test_duplicate_session_identity_with_different_request_is_rejected(schema, file):
    value = fixture(file)
    if schema == "DeletionPreviewRequest":
        value = {"sessions": [{key: row[key] for key in ("session_id", "create_request_id")}
                              for row in value["sessions"]],
                 "analysis_operation_ids": value["analysis_operation_ids"]}
    value["sessions"][1]["session_id"] = value["sessions"][0]["session_id"]
    with pytest.raises(ContractError, match="identities must be unique"):
        validate(schema, value)


@pytest.mark.parametrize("file,schema", [
    ("deletion-accept.json", "AcceptSessionDeletion"),
    ("deletion-analysis-only-preview.json", "ServiceDeletionPreview"),
])
def test_empty_group_never_grants_deletion(file, schema):
    value = fixture(file)
    value["sessions"] = []
    value["analysis_operation_ids"] = []
    with pytest.raises(ContractError, match="at least one"):
        validate(schema, value)


def test_preview_hash_binds_dependency_and_inventory():
    value = fixture("deletion-preview.json")
    value["dependencies"][0]["target_id"] = "00000000-0000-0000-0000-000000000016"
    value["dependencies"][0]["selected"] = False
    with pytest.raises(ContractError, match="dependency hash"):
        validate("ServiceDeletionPreview", value)
    value = fixture("deletion-preview.json")
    value["known_inventory"]["exclusive_bytes"] += 1
    with pytest.raises(ContractError, match="preview hash"):
        validate("ServiceDeletionPreview", value)


def test_never_admitted_revision_cannot_describe_existing_science():
    value = fixture("deletion-absent-preview.json")
    value["sessions"][0]["state"] = "completed"
    with pytest.raises(ContractError, match="never-admitted"):
        validate("ServiceDeletionPreview", value)


@pytest.mark.parametrize("field,replacement", [
    ("deletion_fence", 1), ("effects_started", True),
    ("inventory", {"id": "00000000-0000-0000-0000-000000000021", "sha256": "a" * 64,
                   "file_count": 0, "inode_count": 0, "exclusive_bytes": 0}),
    ("progress", {"removed_files": 1, "removed_bytes": 0, "records_removed": False}),
])
def test_definitive_rejection_proves_zero_admission_and_cleanup(field, replacement):
    value = fixture("deletion-operation-rejected.json")
    value[field] = replacement
    with pytest.raises(ContractError, match="zero admission"):
        validate("ServiceDeletionOperation", value)


def test_quiesced_and_success_receipts_never_overstate_cleanup():
    value = fixture("deletion-operation-quiesced.json")
    value["deletion_fence"] = 0
    with pytest.raises(ContractError, match="durable deletion fence"):
        validate("ServiceDeletionOperation", value)
    value = fixture("deletion-operation-quiesced.json")
    value["inventory"] = None
    with pytest.raises(ContractError, match="exact frozen inventory"):
        validate("ServiceDeletionOperation", value)
    value = fixture("deletion-operation-succeeded.json")
    value["progress"]["removed_bytes"] -= 1
    with pytest.raises(ContractError, match="exact byte"):
        validate("ServiceDeletionOperation", value)


def test_removal_barrier_cannot_substitute_another_fence():
    value = fixture("deletion-removal.json")
    value["public_cleanup_barrier"]["deletion_fence"] += 1
    with pytest.raises(ContractError, match="public barrier differs"):
        validate("CommitSessionRemoval", value)


@pytest.mark.parametrize("stage,resume,field,replacement", [
    ("blocked", "quiesced", "inventory", None),
    ("failed", "removing_outputs", "inventory", None),
    ("quiesced", None, "progress", {"removed_files": 1, "removed_bytes": 0, "records_removed": False}),
    ("blocked", "quiesced", "progress", {"removed_files": 0, "removed_bytes": 1, "records_removed": False}),
    ("quiesced", None, "progress", {"removed_files": 0, "removed_bytes": 0, "records_removed": True}),
    ("removing_outputs", None, "progress", {"removed_files": 0, "removed_bytes": 0, "records_removed": True}),
    ("removing_records", None, "progress", {"removed_files": 0, "removed_bytes": 0, "records_removed": False}),
])
def test_recovery_receipt_cannot_claim_unproven_completed_phase(stage, resume, field, replacement):
    value = fixture("deletion-operation-quiesced.json")
    value.update(stage=stage, resume_stage=resume)
    if stage in {"blocked", "failed"}:
        value["failure"] = {"code": "closure_unknown", "retryable": True}
    value[field] = replacement
    with pytest.raises(ContractError):
        validate("ServiceDeletionOperation", value)


@pytest.mark.parametrize("stage", ["accepted", "quiescing"])
def test_pre_inventory_stages_cannot_claim_output_or_record_removal(stage):
    value = fixture("deletion-operation-quiesced.json")
    value.update(stage=stage, inventory=None)
    value["progress"]["removed_files"] = 1
    with pytest.raises(ContractError, match="zero before output removal"):
        validate("ServiceDeletionOperation", value)


def test_frozen_inventory_totals_cannot_hide_or_invent_owned_inodes():
    value = fixture("deletion-operation-quiesced.json")["inventory"]
    value["inode_count"] = value["file_count"] + 1
    with pytest.raises(ContractError, match="exact file and inode set"):
        validate("FrozenDeletionInventory", value)


def test_deleted_lookup_contains_no_scientific_payload():
    value = fixture("deletion-session-deleted.json")
    value["prepared"] = fixture("session-queued.json")["prepared"]
    with pytest.raises(ContractError):
        validate("DeletedSession", value)


def test_private_body_cannot_claim_owner_authority_or_accept_unordered_group():
    value = fixture("deletion-accept.json")
    value["owner_id"] = "00000000-0000-0000-0000-000000000063"
    with pytest.raises(ContractError):
        validate("AcceptSessionDeletion", value)
    value = fixture("deletion-accept.json")
    value["sessions"].reverse()
    with pytest.raises(ContractError, match="identities must be unique and ordered"):
        validate("AcceptSessionDeletion", value)


def test_private_routes_and_deleted_response_have_one_exact_producer_shape():
    spec = json.loads((FIXTURES.parent / "openapi.json").read_text())
    expected = {
        ("/session-deletions/preview", "post"): ("DeletionPreviewRequest", "ServiceDeletionPreview", "200"),
        ("/session-deletions", "post"): ("AcceptSessionDeletion", "ServiceDeletionOperation", "202"),
        ("/session-deletions/by-request/{request_id}", "get"): (None, "ServiceDeletionOperation", "200"),
        ("/session-deletions/{operation_id}", "get"): (None, "ServiceDeletionOperation", "200"),
        ("/session-deletions/{operation_id}/removal", "post"): ("CommitSessionRemoval", "ServiceDeletionOperation", "202"),
    }
    for (path, method), (request, response, status) in expected.items():
        operation = spec["paths"][path][method]
        assert operation["security"] == [{"ServiceBearer": [], "OwnerScope": []}]
        if request:
            assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/" + request)
        assert operation["responses"][status]["content"]["application/json"]["schema"]["$ref"].endswith("/" + response)
    for path, method in [("/sessions", "post"), ("/sessions/{session_id}", "get"),
                         ("/sessions/by-request/{request_id}", "get")]:
        assert spec["paths"][path][method]["responses"]["410"]["content"]["application/json"]["schema"]["$ref"].endswith("/DeletedSession")
