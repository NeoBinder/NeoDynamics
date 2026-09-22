"""Language-neutral contract fixtures and strict wire conformance."""

import copy
import hashlib
import json
import struct
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
from neomd_server.contract import ContractError, canonical_bytes, content_hash, validate
from neomd_server.preview import PreviewCodec, PreviewFrame

FIXTURES = Path(__file__).parents[2] / "contracts/imd/v1/fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def codec():
    instance = PreviewCodec(fixture("preview-layout.json"))
    instance.acknowledge(fixture("preview-ack.json"))
    return instance


def test_language_neutral_hashes():
    for case in fixture("canonical-hashes.json"):
        raw = canonical_bytes(case["value"])
        assert raw.decode() == case["canonical_utf8"]
        assert raw.hex() == case["canonical_hex"]
        assert content_hash(case["value"]) == case["sha256"]
    assert content_hash({"x": None}) != content_hash({})


@pytest.mark.parametrize(
    "name,schema",
    [
        ("create.json", "CreateRequest"),
        ("command.json", "CommandRequest"),
        ("controller.json", "ControllerRequest"),
        ("session-queued.json", "SessionSnapshot"),
        ('finalized-artifact-page.json', 'FinalizedArtifactPage'),
    ],
)
def test_frozen_requests(name, schema):
    validate(schema, fixture(name))


def test_create_identity_is_part_of_hash():
    request = fixture("create.json")
    request["session_id"] = request["request_id"]
    with pytest.raises(ContractError, match="canonical payload hash"):
        validate("CreateRequest", request)


@pytest.mark.parametrize("replacement", [None, [], 1, {"type": "set_forces", "atom_ids": None}])
def test_malformed_command_collects_protocol_errors(replacement):
    request = fixture("command.json")
    request["payload"] = replacement
    with pytest.raises(ContractError):
        validate("CommandRequest", request)


def test_unknown_paths_units_and_nonfinite_rejected():
    request = fixture("command.json")
    request["payload"]["vectors"][0][0] = float("nan")
    request["payload"]["unit"] = "angstrom"
    request["payload"]["path"] = "/tmp/anything"
    with pytest.raises(ContractError) as error:
        validate("CommandRequest", request)
    assert len(error.value.problems) >= 2


def test_force_count_and_force_norm_validation():
    request = fixture("command.json")["payload"]
    request["vectors"][0] = [10000, 10000, 0]
    request["atom_ids"].pop()
    with pytest.raises(ContractError) as error:
        validate("EngineCommand", request)
    assert len(error.value.problems) == 2


def test_preview_fixture_exact_bytes_and_values():
    expected = fixture("preview-frame.json")
    binary = (FIXTURES / "preview-frame.bin").read_bytes()
    assert len(binary) == expected["bytes"] == 232
    assert hashlib.sha256(binary).hexdigest() == expected["sha256"]
    assert binary.hex() == expected["hex"]
    frame = codec().decode(binary)
    assert frame.sequence == 12 and frame.step == 100 and frame.time_ps == 0.2
    assert frame.positions_nm == tuple(map(tuple, expected["positions_nm"]))
    assert codec().encode(frame) == binary


def test_layout_ack_and_binding_are_mandatory():
    message = fixture("preview-layout.json")
    instance = PreviewCodec(message)
    with pytest.raises(ContractError):
        instance.decode((FIXTURES / "preview-frame.bin").read_bytes())
    bad_ack = fixture("preview-ack.json")
    bad_ack["layout_sha256"] = "0" * 64
    with pytest.raises(ContractError):
        instance.acknowledge(bad_ack)
    message["layout"]["binding"]["generation"] += 1
    message["layout_sha256"] = content_hash(message["layout"])
    different = PreviewCodec(message)
    different.acknowledge({"type": "layout_ack", "layout_sha256": message["layout_sha256"]})
    with pytest.raises(ContractError):
        different.decode((FIXTURES / "preview-frame.bin").read_bytes())


@pytest.mark.parametrize("offset", [0, 10, 12, 16, 24, 56, 60, 64, 96, 209])
def test_corrupted_frames_rejected(offset):
    data = bytearray((FIXTURES / "preview-frame.bin").read_bytes())
    data[offset] ^= 128
    with pytest.raises(ContractError):
        codec().decode(data)


def test_nonfinite_frame_and_reordering_rejected():
    data = bytearray((FIXTURES / "preview-frame.bin").read_bytes())
    struct.pack_into("<f", data, 208, float("nan"))
    data[96:128] = hashlib.sha256(data[208:]).digest()
    with pytest.raises(ContractError, match="nonfinite"):
        codec().decode(data)
    instance = codec()
    binary = (FIXTURES / "preview-frame.bin").read_bytes()
    instance.decode(binary)
    with pytest.raises(ContractError):
        instance.decode(binary)
    with pytest.raises(ContractError):
        codec().decode(binary + b"\0")


def test_null_time_and_box_and_immutable_layout_input():
    message = fixture("preview-layout.json")
    original = copy.deepcopy(message)
    instance = PreviewCodec(message)
    instance.acknowledge(fixture("preview-ack.json"))
    message["layout"]["binding"]["generation"] = 999
    frame = PreviewFrame(0, 0, None, ((0, 0, 0), (1, 1, 1)), None)
    binary = instance.encode(frame)
    receiver = PreviewCodec(original)
    receiver.acknowledge(fixture("preview-ack.json"))
    assert receiver.decode(binary) == frame


@pytest.mark.parametrize(
    "box", [((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((-1, 0, 0), (0, 1, 0), (0, 0, 1))]
)
def test_preview_rejects_degenerate_and_left_handed_cell(box):
    frame = PreviewFrame(1, 1, 0.002, ((0, 0, 0), (1, 1, 1)), box)
    with pytest.raises(ContractError, match="positive volume"):
        codec().encode(frame)
    binary = bytearray((FIXTURES / "preview-frame.bin").read_bytes())
    struct.pack_into("<9d", binary, 128, *(value for row in box for value in row))
    with pytest.raises(ContractError):
        codec().decode(binary)


def test_same_step_frames_carry_monotonic_applied_control_revision():
    from dataclasses import replace

    receiver = codec()
    frame = PreviewFrame(1, 10, 0.02, ((0, 0, 0), (1, 1, 1)), None, control_revision=3)
    assert receiver.decode(codec().encode(frame)).control_revision == 3
    updated = replace(frame, sequence=2, control_revision=4)
    assert receiver.decode(codec().encode(updated)).step == 10
    stale = replace(frame, sequence=3, control_revision=2)
    with pytest.raises(ContractError):
        receiver.decode(codec().encode(stale))
    invalid = replace(frame, sequence=4, control_revision=2**53)
    with pytest.raises(ContractError):
        codec().encode(invalid)


def test_runtime_budget_has_explicit_accounting_and_closed_host_stop():
    value = fixture("runtime-budget.json")
    validate("RuntimeBudget", value)
    for mutate in [lambda v: v.update(elapsed_ms=None),
                   lambda v: v.update(elapsed_ms=2**53),
                   lambda v: v["stop"].update(applied_control_revision=None),
                   lambda v: v["stop"].update(reason="invented")]:
        bad = copy.deepcopy(value)
        mutate(bad)
        with pytest.raises(ContractError):
            validate("RuntimeBudget", bad)
    validate("RuntimeBudget", {"limit_ms": None, "elapsed_ms": None,
                               "accounting": "unknown", "stop": None})


def test_finalization_fixture_closes_states_failures_and_artifact_set():
    value = fixture('finalization.json')
    validate('OutputFinalization', value)
    for mutate in [lambda v: v.update(status='invented'),
                   lambda v: v.update(failure='invented'),
                   lambda v: v.update(total_files=4097),
                   lambda v: v.update(processed_bytes=2**53),
                   lambda v: v.update(status='complete'),
                   lambda v: v.update(artifact_ids=['00000000-0000-4000-8000-000000000001'])]:
        bad = copy.deepcopy(value)
        mutate(bad)
        with pytest.raises(ContractError):
            validate('OutputFinalization', bad)


def test_pre_seal_failure_has_unknown_totals_without_fake_progress():
    value = fixture('finalization-pre-seal-failure.json')
    validate('OutputFinalization', value)
    for mutate in [lambda v: v.update(total_bytes=0), lambda v: v.update(total_files=0),
                   lambda v: v.update(processed_bytes=1), lambda v: v.update(processed_files=1),
                   lambda v: v.update(status='running'), lambda v: v.update(status='complete'),
                   lambda v: v.update(failure=None)]:
        bad = copy.deepcopy(value)
        mutate(bad)
        with pytest.raises(ContractError):
            validate('OutputFinalization', bad)
    validate("Error", {"code": "running_time_limit_exceeded",
                       "message": "running_time_limit_exceeded", "retryable": False})
