"""Mapped, bounded analysis of sealed comparison trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections import defaultdict
from pathlib import Path

import numpy as np

from neomd.analysis.component_comparison import evaluate_component_observables
from neomd.analysis.gamd_frames import read_gamd_frames
from neomd.probes import _STATE_HEADER

from .comparison import _provenance, _selection
from .contract import content_hash, validate
from .results import RESULT_SCHEMA


def _verify_file(path, descriptor, *, max_bytes=100_000_000):
    path = Path(path)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size != descriptor["size_bytes"]
            or before.st_size > max_bytes):
        raise ValueError("sealed source size or type changed")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("sealed source identity changed")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    if (digest.hexdigest() != descriptor["sha256"] or
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("sealed source hash or identity changed")


def _read_state(path, max_rows=1024):
    with Path(path).open(encoding="utf-8") as stream:
        if stream.readline().rstrip("\n") != _STATE_HEADER:
            raise ValueError("unrecognized state report header")
        rows = []
        for line in stream:
            if len(rows) >= max_rows:
                raise ValueError("too many state rows")
            cells = line.rstrip("\n").split("\t")
            if len(cells) != 9:
                raise ValueError("invalid state report row")
            step = int(cells[0])
            time_ps, temperature, volume = map(float, (cells[1], cells[5], cells[6]))
            if (step < 0 or not all(map(math.isfinite, (time_ps, temperature, volume)))
                    or temperature <= 0 or volume <= 0):
                raise ValueError("nonfinite state diagnostics")
            rows.append((step, time_ps, temperature, volume))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate state step")
    return {row[0]: row[1:] for row in rows}


def _measurements(member, definition, paths, manifest, mapping, assembly):
    source = member["source"]
    for key in ("result_manifest", "trajectory", "state"):
        _verify_file(paths[key], source[key])
    result = json.loads(Path(paths["result_manifest"]).read_bytes())
    validate(RESULT_SCHEMA, result)
    if (result["execution"]["terminal_state"] != "completed"
            or result["execution"]["method"] != "prod"
            or result["execution"]["session_id"] != member["session_id"]
            or result["prepared"] != member["prepared"]
            or result["protocol"] != {"protocol_id": member["protocol_id"],
                                      "content_sha256": member["protocol_sha256"]}
            or len(result["trajectories"]) != 1):
        raise ValueError("result manifest scientific identity changed")
    trajectory = result["trajectories"][0]
    if any(trajectory[key] != source["trajectory"][alias] for key, alias in (
            ("artifact_id", "artifact_id"), ("content_sha256", "sha256"),
            ("size_bytes", "size_bytes"))):
        raise ValueError("result DCD identity changed")
    if not trajectory["periodic"] or trajectory["frame_count"] > 256:
        raise ValueError("only bounded periodic DCD is qualified")
    archive = read_gamd_frames(paths["trajectory"], timestep_ps=trajectory["timestep_ps"],
                               masses=[row["mass_dalton"] for row in manifest["particles"]])
    if (len(archive["steps"]) != trajectory["frame_count"]
            or len(archive["steps"]) > 256
            or int(archive["steps"][-1]) != trajectory["last_step"]):
        raise ValueError("DCD frame metadata changed")
    selected = _selection(definition, manifest, mapping, assembly)
    all_indices = [i for group in selected for i in group]
    local = {global_index: index for index, global_index in enumerate(all_indices)}
    source_positions = {
        atom["source_atom_id"]: atom["position_nm"]
        for source_row in manifest["sources"] for atom in source_row["atoms"]
    }
    atom_by_id = {atom["atom_id"]: atom for atom in manifest["atoms"]}
    reference = []
    for global_index in all_indices:
        atom = atom_by_id[manifest["particles"][global_index]["identity_id"]]
        reference.append(source_positions[atom["source_atom_id"]])
    reference = np.asarray(reference, dtype=float)
    by_atom = {row["identity_id"]: row["particle_index"] for row in manifest["particles"]
               if row["identity_kind"] == "atom"}
    bonds = [[local[by_atom[a]], local[by_atom[b]]] for row in manifest["bonds"]
             for a, b in [row["atom_ids"]] if a in by_atom and b in by_atom
             and by_atom[a] in local and by_atom[b] in local]
    local_groups = [[local[i] for i in group] for group in selected]
    state = _read_state(paths["state"])
    measurements = evaluate_component_observables(
        positions_nm=archive["positions_nm"][:, all_indices, :],
        boxes_nm=archive["boxes_nm"], steps=archive["steps"],
        time_ps=archive["time_ps"], groups=local_groups, bonds=bonds,
        reference_nm=reference, state_by_step=state,
        instance_ids=[row["instance_id"] for row in definition["ligands"]],
        discard_frames=definition["equilibration_discard_frames"],
    )
    return measurements, len(all_indices)


def analyse_cohort(request, jobs):
    """All numerical work runs from host verified paths; receipt contains no path."""
    results = []
    for member, job in zip(request["members"], jobs):
        result = {"variant_id": member["variant_id"],
                  "replica_index": member["replica_index"], "session_id": member["session_id"],
                  "source_result_sha256": (member["source"] or {}).get("result_manifest", {}).get("sha256"),
                  "scientific_state": member["scientific_state"],
                  "publication_state": member["publication_state"],
                  "status": "excluded", "reason": job["reason"],
                  "intervention_state": job["intervention_state"],
                  "mapped_atom_count": 0, "measurements": [], "error": None}
        if job["reason"] == "complete":
            try:
                actual, manifest, mapping, assembly = _provenance(
                    job["prepared_directory"], member["prepared"], job["assembly_receipt"])
                if actual["source_content_sha256"] != request["definition"]["source_content_sha256"]:
                    raise ValueError("source chemistry lineage differs")
                values, count = _measurements(member, request["definition"], job["paths"],
                                              manifest, mapping, assembly)
                result.update(status="included", mapped_atom_count=count, measurements=values)
            except Exception:
                result.update(status="failed", reason="analysis_failed",
                              error={"code": "analysis_failed",
                                     "message": "Sealed source or mapped measurement verification failed."})
        results.append(result)
    grouped = defaultdict(list)
    for member in results:
        if member["status"] != "included":
            continue
        for measurement in member["measurements"]:
            key = (member["variant_id"], measurement["observable"], measurement["selector"],
                   measurement["unit"])
            grouped[key].append(measurement["mean"])
    aggregates = []
    for (variant, observable, selector, unit), values in sorted(grouped.items(),
            key=lambda item: str(item[0])):
        count = len(values)
        sd = float(np.std(values, ddof=1)) if count >= 2 else None
        aggregates.append({"variant_id": variant, "observable": observable, "selector": selector,
                           "unit": unit, "replica_count": count, "mean": float(np.mean(values)),
                           "sd": sd, "sem": sd / math.sqrt(count) if sd is not None else None,
                           "confidence_interval": None,
                           "uncertainty_status": "insufficient_replicas" if count < 3 else "descriptive_only"})
    receipt = {"schema_version": 1, "comparison_id": request["comparison_id"],
               "plan_sha256": request["plan_sha256"],
               "definition_sha256": request["definition_sha256"],
               "request_sha256": content_hash(request), "members": results, "aggregates": aggregates}
    validate("urn:neodyn:trajectory-comparison:v1#/$defs/AnalysisReceipt", receipt)
    return receipt
