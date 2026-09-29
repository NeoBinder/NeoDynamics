"""Exact-file child execution at the service-to-SDK component boundary."""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
from pathlib import Path

import pytest
import rfc8785

from neomd_server.component_preparation_execution import (
    _prepare_component_process,
    _validate_component_process,
    _verify_component_process,
)
from neomd_server.ipc import Channel

ROOT = Path(__file__).resolve().parents[2]


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _job(tmp_path, kind, options, files, *, request_id="00000000-0000-4000-8000-000000000003"):
    directory = tmp_path / "inputs"
    directory.mkdir()
    index = {}
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        index[name] = {"sha256": _hash(data), "size_bytes": len(data)}
    index_bytes = rfc8785.dumps(index)
    (directory / "index.json").write_bytes(index_bytes)
    return {"kind": kind, "options": options, "request_id": request_id,
            "input_dir": str(directory), "input_index_sha256": _hash(index_bytes),
            "staging": str(tmp_path / "staging"), "memory_limit_bytes": 20 * 1024**3,
            "max_result_bytes": 512 * 1024**2}


def _child(target, job, *, timeout=180):
    parent, child = Channel.pair(max_message_bytes=1_048_576)
    process = multiprocessing.get_context("spawn").Process(
        target=target, args=(job, child, os.getpid()))
    try:
        process.start()
        child.close()
        outcome = parent.get(timeout=timeout)
        process.join(timeout=10)
        assert process.exitcode == 0
        return outcome
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
        parent.close()
        child.close()


def test_declarative_import_runs_and_staged_tampering_fails_closed(tmp_path):
    fixtures = ROOT / "tests/v2/fixtures/component_import"
    state = json.loads((ROOT / "contracts/component-preparation/v1/fixtures/positive/chemical-state.json")
                       .read_bytes())
    job = _job(tmp_path, "component_import", {"import_mode": "declarative"}, {
        "state.json": rfc8785.dumps(state),
        "manifest.json": (fixtures / "manifest.json").read_bytes(),
        "members/structure.json": (fixtures / "structure.json").read_bytes(),
        "members/parameters.json": (fixtures / "parameters.json").read_bytes(),
    })
    member = Path(job["input_dir"]) / "members/parameters.json"
    original_member = member.read_bytes()
    member.write_bytes(original_member + b" ")
    assert _child(_validate_component_process, job) == {"error": "integrity_mismatch"}
    member.write_bytes(original_member)
    assert _child(_validate_component_process, job) == {"ok": True}
    prepared = _child(_prepare_component_process, job)
    assert prepared["ok"] and prepared["kind"] == "component_import"
    assert prepared["primary_sha256"] == _hash(
        (Path(job["staging"]) / "result/package.json").read_bytes())
    package = Path(job["staging"]) / "result/package.json"
    original = package.read_bytes()
    package.write_bytes(original + b" ")
    assert _child(_verify_component_process, {**job, "prepared": prepared}) == {
        "error": "integrity_mismatch"}
    package.write_bytes(original)
    verified = _child(_verify_component_process, {**job, "prepared": prepared})
    assert verified["ok"] and verified["primary_sha256"] == prepared["primary_sha256"]
    assert verified["report_sha256"] == next(
        row["sha256"] for row in verified["artifacts"]
        if row["path"] == "assets/" + json.loads(original)["qc"]["report"]["artifact_id"])
    assert len(verified["artifacts"]) == len(prepared["artifacts"])


def test_source_bound_review_requires_exact_evidence_and_authorization_hash(tmp_path):
    from tests.v2.test_component_inspection import bind, evidence, source, state

    records = source(copies=1)
    reviewed = state()
    attachments = evidence(records, reviewed)
    instance = bind(records, reviewed)
    authorization = rfc8785.dumps({"decision_id": reviewed["review"]["decision_id"]})
    files = {
        "source.json.gz": gzip.compress(rfc8785.dumps(records), mtime=0),
        "state.json": rfc8785.dumps(reviewed),
        "instance.json": rfc8785.dumps(instance),
        "review-authorization.json": authorization,
        **{f"evidence/{digest}": data for digest, data in attachments.items()},
    }
    job = _job(tmp_path, "component_review",
               {"review_authorization_sha256": _hash(authorization)}, files)
    assert _child(_validate_component_process, job) == {"ok": True}
    assert _child(_validate_component_process, {
        **job, "options": {"review_authorization_sha256": "0" * 64}}) == {
            "error": "evidence_mismatch"}
    prepared = _child(_prepare_component_process, job)
    verified = _child(_verify_component_process, {**job, "prepared": prepared})
    assert verified["ok"] and verified["primary_sha256"] == _hash(files["state.json"])
    assert verified["report_sha256"] == next(
        row["sha256"] for row in verified["artifacts"] if row["path"] == "review-report.json")


def test_sep_generate_child_uses_exact_reviewed_chain_and_publishes_assets(tmp_path):
    configured = os.environ.get("NEOMD_P11_SOURCE_DIR")
    if not configured:
        pytest.skip("set NEOMD_P11_SOURCE_DIR for retained 2MZ2 SEP operation")
    folder = Path(configured)
    files = {name: (folder / name).read_bytes() for name in
             ("source.json.gz", "accepted-state.json", "instance.json")}
    options = {"method_id": "amber-ff14sb-phosaa14sb-sep",
               "source_residue_id": "2", "template_id": "SEP",
               "template_resource_sha256":
               "991d13246deebe35e362ad8d99c589a4a2fcf7322c97a408578f7c8c874ae08b"}
    job = _job(tmp_path, "component_generate", options, {
        "source.json.gz": files["source.json.gz"],
        "state.json": files["accepted-state.json"],
        "instance.json": files["instance.json"],
    })
    assert _child(_validate_component_process, job) == {"ok": True}
    prepared = _child(_prepare_component_process, job)
    assert prepared["ok"] and prepared["kind"] == "component_generate"
    package = json.loads((Path(job["staging"]) / "result/package.json").read_bytes())
    assert package["method"]["method_id"] == "openmm.ff14sb.phosaa14sb.sep"
    assert len(package["assets"]) == 7
    assert package["charge_sum_e"] == 0
    verified = _child(_verify_component_process, {**job, "prepared": prepared})
    assert verified["ok"] and verified["primary_sha256"] == prepared["primary_sha256"]
    assert len(verified["artifacts"]) == 10
    assert _child(_validate_component_process, {
        **job, "options": {**options, "source_residue_id": "3"}}) == {
            "error": "capability_unavailable"}
    assert _child(_validate_component_process, {
        **job, "options": {**options, "template_resource_sha256": "0" * 64}}) == {
            "error": "capability_unavailable"}


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_gaff_generation_publishes_exact_qc_asset(tmp_path):
    reviewed = json.loads((ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json").read_bytes())
    state = reviewed["chemical_state"]
    job = _job(tmp_path, "component_generate", {"method_id": "gaff2", "seed": 20260926},
               {"state.json": rfc8785.dumps(state)})
    assert _child(_validate_component_process, job) == {"ok": True}
    prepared = _child(_prepare_component_process, job, timeout=240)
    assert prepared["ok"] and prepared["qc"]["asset_count"] == 10
    verified = _child(_verify_component_process, {**job, "prepared": prepared})
    assert verified["ok"] and verified["primary_sha256"] == prepared["primary_sha256"]
    package = json.loads((Path(job["staging"]) / "result/package.json").read_bytes())
    assert verified["report_sha256"] == next(
        row["sha256"] for row in verified["artifacts"]
        if row["path"] == "assets/" + package["qc"]["report"]["artifact_id"])


@pytest.mark.skipif(shutil.which("antechamber") is None, reason="requires locked AmberTools")
def test_sage_generation_runs_in_isolated_child_and_verifies_exact_assets(tmp_path):
    reviewed = json.loads((ROOT / "tests/v2/fixtures/3htb_jz4_reviewed.json").read_bytes())
    job = _job(tmp_path, "component_generate",
               {"method_id": "openff-sage-2.2.1", "seed": 20260926},
               {"state.json": rfc8785.dumps(reviewed["chemical_state"])})
    assert _child(_validate_component_process, job) == {"ok": True}
    prepared = _child(_prepare_component_process, job, timeout=300)
    assert prepared["ok"] and prepared["qc"]["asset_count"] == 10
    verified = _child(_verify_component_process, {**job, "prepared": prepared})
    assert verified["ok"] and verified["primary_sha256"] == prepared["primary_sha256"]
    package = json.loads((Path(job["staging"]) / "result/package.json").read_bytes())
    assert package["method"]["method_id"] == "openff.sage2.2.1.am1bcc"
    assert verified["report_sha256"] == next(
        row["sha256"] for row in verified["artifacts"]
        if row["path"] == "assets/" + package["qc"]["report"]["artifact_id"])


@pytest.mark.skipif(not os.environ.get("NEOMD_P10_CAPTURE_ROOT"),
                    reason="requires retained source-bound P10 task capture")
def test_sage_two_copy_assembly_child_publishes_prepared_v2(tmp_path):
    from neomd.component_assembly import _profile
    from neomd.component_contract import ComponentContractError

    capture = Path(os.environ["NEOMD_P10_CAPTURE_ROOT"])
    source = capture / "P09-canova-exact-assembly-inputs"
    sage = capture / "P10-jz4-openff-package"
    assembly = capture / "P10-jz4-openff-assembly"
    state = json.loads((sage / "state.json").read_bytes())
    package = json.loads((sage / "package.json").read_bytes())
    recipe = json.loads((assembly / "recipe.json").read_bytes())
    gaff = json.loads(next((source / "packages").glob("*.json")).read_bytes())
    with pytest.raises(ComponentContractError, match="mixed ligand families"):
        _profile(recipe, [package, gaff])
    sid, pid = state["state_id"], package["package_id"]
    files = {"source.json.gz": (source / "source.json.gz").read_bytes(),
             "recipe.json": rfc8785.dumps(recipe),
             f"states/{sid}.json": rfc8785.dumps(state),
             f"packages/{pid}.json": rfc8785.dumps(package)}
    for path in (source / "instances").glob("*.json"):
        files[f"instances/{path.stem}.json"] = path.read_bytes()
    for path in (source / "poses").glob("*.json"):
        files[f"poses/{path.stem}.json"] = path.read_bytes()
    for path in (sage / "assets").iterdir():
        files[f"assets/{pid}/{path.name}"] = path.read_bytes()
    job = _job(tmp_path, "component_assemble", {"mode": "stored-package-v1"}, files)
    assert _child(_validate_component_process, job) == {"ok": True}
    prepared = _child(_prepare_component_process, job, timeout=300)
    assert prepared["ok"] and prepared["qc"]["particle_count"] == 6131
    verified = _child(_verify_component_process, {**job, "prepared": prepared},
                      timeout=300)
    assert verified["ok"] and verified["qc"]["status"] == "numerically_executable"
    assert verified["result_sha256"] == verified["qc"]["prepared_manifest_sha256"]
    assert verified["report_sha256"] == verified["qc"]["qc_report_sha256"]
    from neomd.prepared import read_prepared_bundle

    result = Path(job["staging"]) / "result"
    reopened = read_prepared_bundle(result / "prepared/manifest.json",
                                    expected_sha256=verified["result_sha256"])
    assert reopened["particle_count"] == 6131


def test_full_3htb_source_absent_pose_atoms_fail_without_host_provenance(tmp_path):
    from tests.v2.test_component_assembly_full_3htb import load_full_3htb_assembly_inputs

    source, _, states, instances, packages, recipe, assets, poses, receipts = (
        load_full_3htb_assembly_inputs())
    files = {"source.json.gz": source, "recipe.json": rfc8785.dumps(recipe)}
    for state, instance, package in zip(states, instances, packages):
        sid, iid, pid = state["state_id"], instance["instance_id"], package["package_id"]
        files[f"states/{sid}.json"] = rfc8785.dumps(state)
        files[f"instances/{iid}.json"] = rfc8785.dumps(instance)
        files[f"packages/{pid}.json"] = rfc8785.dumps(package)
        files[f"poses/{iid}.json"] = rfc8785.dumps(poses[iid])
        for aid, data in assets[pid].items():
            files[f"assets/{pid}/{aid}"] = data
        if iid in receipts:
            files[f"pose-receipts/{iid}.json"] = receipts[iid]["receipt_bytes"]
    job = _job(tmp_path, "component_assemble", {"mode": "stored-package-v1"}, files)
    assert _child(_validate_component_process, job) == {"error": "capability_unavailable"}
    assert not (Path(job["staging"]) / "result").exists()
    for name in list(files):
        if name.startswith("pose-receipts/"):
            del files[name]
    second = tmp_path / "without-receipt"
    second.mkdir()
    job = _job(second, "component_assemble", {"mode": "stored-package-v1"}, files)
    assert _child(_validate_component_process, job) == {"error": "capability_unavailable"}
    assert not (Path(job["staging"]) / "result").exists()
