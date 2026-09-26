"""Isolated child-process targets for preparation validation, execution and verification.

Every blocking scientific stage runs in a spawned, parent-death-armed child with an
explicit address-space limit. Only compact public-safe results cross the channel;
artifact bytes stay in host-owned staging directories and never enter IPC.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from uuid import UUID

from .ledger import LedgerError
from .processes import arm_parent_death

SCHEMA = "urn:neodyn:protein-preparation:v1:operation"


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _address_limit(memory_bytes):
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    wanted = int(memory_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (wanted, max(wanted, soft) if soft > 0 else wanted))


def _source_file(root, job):
    path = Path(root) / "preparation-uploads" / str(job["upload_id"]) / f"source.{job['format']}"
    if not path.is_file():
        raise LedgerError("integrity_mismatch")
    return path


def load_explicit_topology_source(job, *, require_complete=False):
    """Load an owner-authorized v2 source in the isolated scientific child.

    ``job`` must come from a scoped upload lookup, not a client path. The file
    name is fixed by the format identity and the upload UUID; immutable content,
    decoded size and revision are checked by the SDK before OpenMM sees records.
    """
    from neomd.topology_intake import load_topology_artifact

    upload_id = UUID(str(job["upload_id"]))
    reference = job["reference"]
    path = (Path(job["root"]) / "preparation-uploads" / str(upload_id) /
            "source.molecular-topology-v1+gzip")
    if not path.is_file():
        raise LedgerError("integrity_mismatch")
    return load_topology_artifact(path, reference, require_complete=require_complete)


def _load_source(job):
    from neomd.preparation import inspect_protein, load_protein_source

    path = _source_file(job["root"], job)
    if _digest(path) != job["canonical_bytes_sha256"]:
        raise LedgerError("integrity_mismatch")
    source = load_protein_source(path, job["binding"])
    return source, inspect_protein(source)


def _blockers(source, inspection):
    """Pipeline-faithful acceptance: grouping removes retained HOH before repair,
    so unsupported_residue findings on water chains are not blockers."""
    names = {r["source_residue_id"]: r["name"] for r in source.descriptor["residue_bindings"]}
    return [f for f in inspection["findings"]
            if f["code"] != "sequence_unknown"
            and not (f["code"] == "unsupported_residue"
                     and names.get(f["source_residue_id"]) == "HOH")]


def _validation_errors(error, *, stage):
    """Map collected SDK diagnostics to closed public codes."""
    from neomd.errors import NeoUserError

    problems = getattr(error, "errors", [error] if isinstance(error, NeoUserError) else [])
    messages = [str(getattr(problem, "message", problem)).lower() for problem in problems]
    keys = [str(getattr(problem, "key", "")).lower() for problem in problems]
    text = " ".join(messages + keys)
    if "protonation" in text or "hydrogenation" in text:
        return "protonation_conflict"
    if "unsupported" in text or "grouping" in text:
        return "unsupported_structure"
    return "evidence_mismatch" if stage == "load" else "verification_failed"


def _validate_source_process(job, result, parent_pid):
    arm_parent_death(parent_pid)
    try:
        _address_limit(job["memory_limit_bytes"])
        source, inspection = _load_source(job)
        from neomd.protein_grouping import group_protein_source

        group_protein_source(source)  # Exact partition gate: rejects non-HOH residues early.
        outcome = {"ok": not _blockers(source, inspection),
                   "can_repair": inspection["can_repair"],
                   "blockers": _blockers(source, inspection)}
    except LedgerError as error:
        outcome = {"error": error.code}
    except Exception as error:
        outcome = {"error": _validation_errors(error, stage="load")}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _prepare_process(job, result, parent_pid):
    """Real bounded SDK preparation; publishes one staging bundle or reports a closed code."""
    arm_parent_death(parent_pid)
    log_path = Path(job["staging"]) / "preparation.log"

    def log(line):
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    started = time.monotonic()
    try:
        _address_limit(job["memory_limit_bytes"])
        Path(job["staging"]).mkdir(parents=True, exist_ok=True)
        log(f"stage=load upload_id={job['upload_id']}")
        source, inspection = _load_source(job)
        if _blockers(source, inspection):
            raise LedgerError("unsupported_structure")
        log("stage=prepare begin")
        from neomd.protein_pipeline import prepare_protein_system

        bundle = prepare_protein_system(
            source, job["staging"],
            parameters=job["system_parameters"], pH=job["parameters"]["ph"],
            padding_nm=job["parameters"]["padding_nm"],
            ionic_strength_mol_per_l=job["parameters"]["ionic_strength_mol_per_l"],
            platform="CPU", seed=job["seed"],
        )
        log("stage=prepare published")
        manifest = Path(bundle.prepared_manifest)
        directory = manifest.parent
        artifacts = []
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            relative = path.relative_to(directory).as_posix()
            artifacts.append({"path": relative, "sha256": _digest(path),
                              "size_bytes": path.lstat().st_size})
        total = sum(item["size_bytes"] for item in artifacts)
        if total > job["max_result_bytes"]:
            raise LedgerError("result_too_large")
        import openmm

        outcome = {
            "ok": True,
            "prepared_revision_id": directory.name,
            "manifest_sha256": _digest(manifest),
            "artifacts": artifacts,
            "total_size_bytes": total,
            "wall_time_seconds": round(time.monotonic() - started, 3),
            "openmm_build": openmm.version.full_version,
        }
        log("stage=complete")
    except LedgerError as error:
        outcome = {"error": error.code}
    except MemoryError:
        outcome = {"error": "preparation_memory_limit_exceeded"}
    except Exception as error:
        outcome = {"error": _validation_errors(error, stage="prepare")}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _versions():
    import importlib.metadata

    names = ["NeoDynamics", "numpy", "openff-toolkit", "openmm", "openmmforcefields", "pdbfixer"]
    return {name: importlib.metadata.version(name) for name in names}


def _qc_from_manifest(manifest):
    """Compact public-safe QC summary from the independently verified manifest."""
    atoms = manifest["atoms"]
    steps = {step["operation"]: step for step in manifest["preparation_steps"]}
    solvation = steps["add_solvent"]["solvation"]
    hydrogenation = steps["add_hydrogens"]
    repair = steps["repair_standard_residue"]
    origins = [a["origin"] for a in atoms]
    return {
        "source_atom_count": sum(len(s["atoms"]) for s in manifest["sources"]),
        "retained_source_atom_count": sum(1 for a in origins if a == "source"),
        "generated_heavy_atom_count": sum(1 for a in origins if a == "generated_heavy_atom"),
        "generated_hydrogen_count": sum(1 for a in origins if a == "generated_hydrogen"),
        "retained_water_count": solvation["retained_water_count"],
        "generated_water_count": solvation["generated_water_count"],
        "salt_pair_count": solvation["salt_pair_count"],
        "neutralizing_ion_count": solvation["neutralizing_ion_count"],
        "positive_ion_count": solvation["positive_ion_count"],
        "negative_ion_count": solvation["negative_ion_count"],
        "solute_charge_e": solvation["solute_charge_e"],
        "residue_variant_count": len(hydrogenation.get("residue_variants", [])),
        "added_bond_count": len(repair["repair"]["added_bonds"]) + sum(
            len(step[key]["added_bonds"]) for step, key in
            ((steps["add_hydrogens"], "hydrogenation"), (steps["add_solvent"], "solvation"))),
        "removed_bond_count": sum(len(step.get("repair", {}).get("removed_bonds", []))
                                  for step in steps.values()),
        "particle_count": manifest["particle_count"],
        "box_vectors_nm": manifest["box_vectors_nm"],
    }


def _verify_process(job, result, parent_pid):
    """Independent full-history and System verification of the staged bundle."""
    arm_parent_death(parent_pid)
    try:
        _address_limit(job["memory_limit_bytes"])
        from neomd.prepared_pipeline import verify_protein_preparation

        manifest_path = Path(job["staging"]) / job["prepared_revision_id"] / "manifest.json"
        if _digest(manifest_path) != job["manifest_sha256"]:
            raise LedgerError("integrity_mismatch")
        verified = verify_protein_preparation(manifest_path, expected_sha256=job["manifest_sha256"])
        mapping_sha = verified["artifacts"]["mapping.json"]["sha256"]
        report_sha = verified["artifacts"]["assembly-report.json"]["sha256"]
        qc = _qc_from_manifest(verified)
        outcome = {
            "ok": True,
            "prepared_revision_id": verified["prepared_revision_id"],
            "mapping_sha256": mapping_sha,
            "assembly_report_sha256": report_sha,
            "particle_count": verified["particle_count"],
            "versions": _versions(),
            "qc": qc,
        }
    except LedgerError as error:
        outcome = {"error": error.code}
    except MemoryError:
        outcome = {"error": "preparation_memory_limit_exceeded"}
    except Exception as error:
        outcome = {"error": _validation_errors(error, stage="verify")}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _validate_topology_source_process(job, result, parent_pid):
    """Admit only a complete immutable source before creating operation intent."""
    arm_parent_death(parent_pid)
    try:
        _address_limit(job["memory_limit_bytes"])
        load_explicit_topology_source(job, require_complete=True)
        outcome = {"ok": True}
    except LedgerError as error:
        outcome = {"error": error.code}
    except Exception as error:
        from neomd.topology_adapter import ConnectivityPrerequisiteError
        from neomd.topology_records import TopologyRecordError

        if isinstance(error, ConnectivityPrerequisiteError):
            outcome = {"error": "connectivity_incomplete"}
        elif isinstance(error, TopologyRecordError):
            outcome = {"error": "invalid_topology_record"}
        else:
            outcome = {"error": "evidence_mismatch"}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _prepare_topology_process(job, result, parent_pid):
    """Full explicit-bond ff14SB/TIP3P/GAFF preparation in an isolated child."""
    arm_parent_death(parent_pid)
    started = time.monotonic()
    try:
        _address_limit(job["memory_limit_bytes"])
        from neomd.topology_preparation import prepare_explicit_topology

        staging = Path(job["staging"])
        staging.mkdir(parents=True, exist_ok=True)
        source = _source_file(job["root"], job)
        bundle = prepare_explicit_topology(
            source, job["reference"], staging,
            ph=job["parameters"]["ph"],
            ionic_strength_molar=job["parameters"]["ionic_strength_mol_per_l"],
            padding_nm=job["parameters"]["padding_nm"],
        )
        manifest = Path(bundle.prepared_manifest)
        directory = manifest.parent
        artifacts = [{"path": path.relative_to(directory).as_posix(),
                      "sha256": _digest(path), "size_bytes": path.stat().st_size}
                     for path in sorted(directory.rglob("*")) if path.is_file()]
        total = sum(item["size_bytes"] for item in artifacts)
        if total > job["max_result_bytes"]:
            raise LedgerError("result_too_large")
        import openmm

        outcome = {"ok": True, "prepared_revision_id": directory.name,
                   "manifest_sha256": _digest(manifest), "artifacts": artifacts,
                   "total_size_bytes": total,
                   "wall_time_seconds": round(time.monotonic() - started, 3),
                   "openmm_build": openmm.version.full_version}
    except LedgerError as error:
        outcome = {"error": error.code}
    except MemoryError:
        outcome = {"error": "preparation_memory_limit_exceeded"}
    except Exception as error:
        from neomd.topology_adapter import ConnectivityPrerequisiteError
        from neomd.topology_records import TopologyRecordError

        if isinstance(error, ConnectivityPrerequisiteError):
            outcome = {"error": "connectivity_incomplete"}
        elif isinstance(error, TopologyRecordError):
            outcome = {"error": "invalid_topology_record"}
        else:
            outcome = {"error": _validation_errors(error, stage="prepare")}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _verify_topology_process(job, result, parent_pid):
    """Reopen the exact bundle and verify source, additions and System size."""
    arm_parent_death(parent_pid)
    try:
        _address_limit(job["memory_limit_bytes"])
        from neomd.topology_preparation import verify_explicit_topology_preparation

        manifest_path = (Path(job["staging"]) / "prepared" /
                         job["prepared_revision_id"] / "manifest.json")
        manifest = verify_explicit_topology_preparation(
            manifest_path, expected_sha256=job["manifest_sha256"])
        changes = json.loads(
            (manifest_path.parent / "topology-changes.json").read_text())
        outcome = {
            "ok": True, "prepared_revision_id": manifest["prepared_revision_id"],
            "mapping_sha256": manifest["artifacts"]["source-topology-map.json"]["sha256"],
            "changes_sha256": manifest["artifacts"]["topology-changes.json"]["sha256"],
            "particle_count": manifest["particle_count"],
            "versions": _versions(),
            "qc": {
                "source_atom_count": len(manifest["sources"][0]["atoms"]),
                "retained_source_bond_count": changes["source_bond_count"],
                "added_hydrogen_bond_count": len(changes["added_hydrogen_bonds"]),
                "added_solvation_bond_count": len(changes["added_solvation_bonds"]),
                "particle_count": manifest["particle_count"],
                "box_vectors_nm": manifest["box_vectors_nm"],
            },
        }
    except LedgerError as error:
        outcome = {"error": error.code}
    except MemoryError:
        outcome = {"error": "preparation_memory_limit_exceeded"}
    except Exception:
        outcome = {"error": "verification_failed"}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()
