"""Owner-fenced comparison plan attestation and durable trajectory analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from uuid import UUID, uuid4

from .assets import _artifact_bytes
from .contract import content_hash, validate
from .ledger import LedgerError
from .processes import arm_parent_death

SCHEMA = "urn:neodyn:trajectory-comparison:v1"
CONTROLS = (
    "force_field_package", "chemical_state", "component_charge", "system_charge",
    "protein", "water_settings", "water_count", "water_realization",
    "ion_settings", "ion_count", "ion_realization",
    "box", "constraints", "starting_poses",
)


def _uuid(value):
    return UUID(str(value))


def _verified_json(root, manifest, name):
    return json.loads(_artifact_bytes(root, manifest, name))


def _protein_identity(source, manifest):
    """Identify prepared polymer residues even when sequence evidence is unknown."""
    sequences = source["polymer_sequences"]
    chain_indices = [row["input_chain_index"] for row in sequences]
    if len(chain_indices) != len(set(chain_indices)):
        raise LedgerError("evidence_mismatch")
    polymer_rows = [row for row in source["residue_bindings"]
                    if row["input_chain_index"] in chain_indices]
    if sequences and (not polymer_rows or len({row["source_residue_id"] for row in polymer_rows})
                      != len(polymer_rows)):
        raise LedgerError("evidence_mismatch")
    prepared_residues = defaultdict(list)
    for row in manifest["residues"]:
        prepared_residues[row["source_residue_id"]].append(row)
    atoms = defaultdict(list)
    for row in manifest["atoms"]:
        atoms[row["residue_id"]].append(row)
    records = []
    for source_row in sorted(polymer_rows, key=lambda row: (
            row["input_chain_index"], row["input_residue_index"])):
        matches = prepared_residues[source_row["source_residue_id"]]
        if len(matches) != 1:
            raise LedgerError("evidence_mismatch")
        prepared = matches[0]
        records.append({
            "source": source_row,
            "prepared": {key: prepared[key] for key in (
                "name", "chain_label", "sequence_label", "insertion_code")},
            "atoms": sorted(({key: atom[key] for key in (
                "name", "element", "formal_charge", "origin", "source_atom_id")}
                for atom in atoms[prepared["residue_id"]]), key=lambda row: (
                    row["name"], row["element"], str(row["source_atom_id"]))),
        })
    return len(records), content_hash({"sequences": sequences, "residues": records})


def _realization(assembly, coordinates, origins):
    """Hash exact solvent molecules/ions by verified atom identity and xyz."""
    grouped = defaultdict(list)
    for row in assembly["particles"]:
        if row["origin"] in origins:
            index = row["particle_index"]
            if index < 0 or index >= len(coordinates):
                raise LedgerError("evidence_mismatch")
            grouped[(row["origin"], row["residue_index"])].append({
                "name": row["name"], "element": row["element"],
                "position_nm": list(coordinates[index]),
            })
    molecules = [{"origin": origin, "atoms": sorted(rows, key=lambda row: (
        row["name"], row["element"], row["position_nm"]))}
        for (origin, _residue_index), rows in grouped.items()]
    molecules.sort(key=content_hash)
    return len(molecules), content_hash(molecules)


def _provenance(root, prepared, assembly_receipt):
    """Read only exact prepared/assembly bytes; derive values, never trust labels."""
    root = Path(root).resolve(strict=True)
    raw_manifest = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != prepared["manifest_sha256"]:
        raise LedgerError("integrity_mismatch")
    manifest = json.loads(raw_manifest)
    if (manifest["prepared_revision_id"] != prepared["prepared_revision_id"]
            or manifest["artifacts"]["mapping.json"]["sha256"] != prepared["mapping_sha256"]
            or assembly_receipt["result"] != {
                "id": assembly_receipt["result"]["id"],
                "object_type": "prepared_bundle", "sha256": prepared["manifest_sha256"],
            }):
        raise LedgerError("evidence_mismatch")
    mapping = _verified_json(root, manifest, "mapping.json")
    report = _verified_json(root, manifest, "assembly/report.json")
    assembly = _verified_json(root, manifest, "assembly/manifest.json")
    raw_coordinates = _artifact_bytes(root, manifest, "coordinates.f64le")
    if len(raw_coordinates) != 24 * manifest["particle_count"]:
        raise LedgerError("evidence_mismatch")
    coordinates = list(struct.iter_unpack("<ddd", raw_coordinates))
    if not all(math.isfinite(value) for xyz in coordinates for value in xyz):
        raise LedgerError("evidence_mismatch")
    xml = _artifact_bytes(root, manifest, "system.xml")
    system = ET.fromstring(xml)
    forces = [f for f in system.findall(".//Force") if f.get("type") == "NonbondedForce"]
    if len(forces) != 1:
        raise LedgerError("evidence_mismatch")
    charge_rows = forces[0].findall("Particles/Particle")
    if len(charge_rows) != manifest["particle_count"]:
        raise LedgerError("evidence_mismatch")
    charges = [float(row.attrib["q"]) for row in charge_rows]
    if not all(math.isfinite(q) for q in charges):
        raise LedgerError("evidence_mismatch")
    assigned = report["assignments"]
    component_indices = [i for row in assigned for i in row["particle_indices"]]
    if len(component_indices) != len(set(component_indices)):
        raise LedgerError("evidence_mismatch")
    component_charge = sum(charges[i] for i in component_indices)
    system_charge = sum(charges)
    if (abs(system_charge - report["net_charge_e"]) > 1e-5
            or report["source_content_sha256"] != assembly["source_content_sha256"]
            or report["source_content_sha256"] != manifest["parameters"]["source_content_sha256"]
            or report["box_vectors_nm"] != manifest["box_vectors_nm"]):
        raise LedgerError("evidence_mismatch")
    recipe = report["assembly"]
    solvent = report["solvent"]
    source = manifest["sources"]
    if len(source) != 1 or source[0]["content_sha256"] != report["source_content_sha256"]:
        raise LedgerError("evidence_mismatch")
    source_atoms = source[0]["atoms"]
    source_pose = sorted((row["source_atom_id"], row["position_nm"]) for row in source_atoms)
    chemical_state = {
        "source_content_sha256": report["source_content_sha256"],
        "assigned_states": sorted((row["instance_id"], row["state_id"]) for row in assigned),
        "formal_charges": sorted((row["source_atom_id"], row["formal_charge"]) for row in source_atoms),
    }
    water_settings = {key: recipe[key] for key in (
        "water_model", "padding_nm", "retained_water_policy", "solvation_seed", "box_policy"
    )}
    ion_settings = {key: recipe[key] for key in (
        "neutralize", "ionic_strength_mol_per_l", "positive_ion", "negative_ion"
    )}
    protein_residue_count, protein_identity_sha256 = _protein_identity(source[0], manifest)
    water_count, water_realization_sha256 = _realization(
        assembly, coordinates, {"retained_water", "generated_water"})
    ion_count, ion_realization_sha256 = _realization(
        assembly, coordinates, {"generated_ion"})
    if (water_count != solvent["generated_water"] + solvent["retained_water"]
            or ion_count != solvent["sodium"] + solvent["chloride"]):
        raise LedgerError("evidence_mismatch")
    actual = {
        "source_content_sha256": report["source_content_sha256"],
        "package_sha256": content_hash(report["package_sha256"]),
        "chemical_state_sha256": content_hash(chemical_state),
        "component_charge_e": component_charge,
        "system_charge_e": system_charge,
        "protein_residue_count": protein_residue_count,
        "protein_identity_sha256": protein_identity_sha256,
        "water_model": recipe["water_model"],
        "water_settings_sha256": content_hash(water_settings),
        "water_count": water_count,
        "water_realization_sha256": water_realization_sha256,
        "ion_count": ion_count,
        "ion_realization_sha256": ion_realization_sha256,
        "ion_settings_sha256": content_hash(ion_settings),
        "ionic_strength_molar": recipe["ionic_strength_mol_per_l"],
        "box_vectors_nm": manifest["box_vectors_nm"],
        "constraints": recipe["constraints"],
        "hydrogen_mass_amu": recipe["hydrogen_mass_amu"],
        "source_pose_sha256": content_hash(source_pose),
        "particle_count": manifest["particle_count"],
        "nonbonded_method": recipe["nonbonded_method"],
    }
    validate(f"{SCHEMA}#/$defs/ActualProvenance", actual)
    return actual, manifest, mapping, assembly


def _selection(definition, manifest, mapping, assembly):
    """Resolve semantic source atom IDs through each prepared revision separately."""
    if definition["source_content_sha256"] != assembly["source_content_sha256"]:
        raise LedgerError("evidence_mismatch")
    atom_by_id = {row["atom_id"]: row for row in manifest["atoms"]}
    particle_by_index = {row["particle_index"]: row for row in manifest["particles"]}
    mapped_atoms = [
        {key: atom[key] for key in ("atom_id", "residue_id", "source_id", "source_atom_id")}
        for atom in manifest["atoms"]
    ]
    mapped_particles = {row["particle_index"]: row for row in mapping["particles"]}
    if (mapping["atoms"] != mapped_atoms
            or len(particle_by_index) != manifest["particle_count"]
            or len(mapped_particles) != manifest["particle_count"]
            or any(mapped_particles[i]["identity_id"] != row["identity_id"]
                   for i, row in particle_by_index.items())):
        raise LedgerError("evidence_mismatch")
    lineage = {}
    for particle in assembly["particles"]:
        if particle["origin"] != "component":
            continue
        index = particle["particle_index"]
        atom = atom_by_id.get(mapped_particles[index]["identity_id"])
        if atom is None or atom["source_atom_id"] is None:
            raise LedgerError("evidence_mismatch")
        key = (particle["instance_id"], atom["source_atom_id"])
        if key in lineage:
            raise LedgerError("evidence_mismatch")
        lineage[key] = (index, atom["element"])
    selected = []
    for ligand in definition["ligands"]:
        indices = []
        available_heavy = {source_atom_id for (instance_id, source_atom_id), value in lineage.items()
                           if instance_id == ligand["instance_id"] and value[1] != "H"}
        if set(ligand["source_atom_ids"]) != available_heavy:
            raise LedgerError("evidence_mismatch")
        for source_atom_id in ligand["source_atom_ids"]:
            row = lineage.get((ligand["instance_id"], source_atom_id))
            if row is None or row[1] == "H":
                raise LedgerError("evidence_mismatch")
            indices.append(row[0])
        if len(indices) != len(set(indices)) or len(indices) < 3:
            raise LedgerError("evidence_mismatch")
        selected.append(indices)
    if len(set(i for group in selected for i in group)) != sum(map(len, selected)):
        raise LedgerError("evidence_mismatch")
    return selected


def _control_value(name, actual):
    return {
        "force_field_package": actual["package_sha256"],
        "chemical_state": actual["chemical_state_sha256"],
        "component_charge": round(actual["component_charge_e"], 6),
        "system_charge": round(actual["system_charge_e"], 6),
        "protein": [actual["protein_residue_count"], actual["protein_identity_sha256"]],
        "water_settings": actual["water_settings_sha256"],
        "water_count": actual["water_count"],
        "water_realization": actual["water_realization_sha256"],
        "ion_settings": actual["ion_settings_sha256"],
        "ion_count": actual["ion_count"],
        "ion_realization": actual["ion_realization_sha256"],
        "box": actual["box_vectors_nm"],
        "constraints": [actual["constraints"], actual["hydrogen_mass_amu"]],
        "starting_poses": actual["source_pose_sha256"],
    }[name]


def _compare_controls(variants, actual_rows):
    controls = []
    for name in CONTROLS:
        digests = [content_hash(_control_value(name, v["actual"])) for v in actual_rows]
        status = "held_fixed" if len(set(digests)) == 1 else "differs"
        if status == "differs" and any(name in v["provenance"]["held_fixed"] for v in variants):
            raise LedgerError("evidence_mismatch")
        controls.append({"field": name, "status": status, "values_sha256": digests})
    net_charges = [
        (round(v["actual"]["component_charge_e"], 6),
         round(v["actual"]["system_charge_e"], 6))
        for v in actual_rows
    ]
    if len(set(net_charges)) > 1 and any(
            "ion_decision" not in v["provenance"] for v in variants):
        raise LedgerError("evidence_mismatch")
    return controls


def _check_slot_binding(slot, prepared, template_sha256, registered, binding):
    if (registered is None or registered["binding"] != prepared
            or registered["method"] != "prod"
            or registered["content_sha256"] != slot["protocol_sha256"]
            or registered["descriptor"]["physics"]["seed"] != slot["seed"]
            or binding["status"] != "ready"
            or binding["template_sha256"] != template_sha256
            or binding["protocol"] != registered["descriptor"]
            or binding["registration_sha256"] != slot["protocol_sha256"]):
        raise LedgerError("evidence_mismatch")


class ComparisonStore:
    def __init__(self, ledger, inputs):
        self.ledger, self.inputs = ledger, inputs
        self._tasks = set()

    async def validate_plan(self, scope_id, plan):
        validate(f"{SCHEMA}#/$defs/Plan", plan)
        if (content_hash(plan["template"]) != plan["template_sha256"]
                or plan["template"]["protocol"]["method"] != plan["method"]):
            raise LedgerError("evidence_mismatch")
        variants = plan["variants"]
        variant_ids = [v["variant_id"] for v in variants]
        if len(set(variant_ids)) != len(variants):
            raise LedgerError("invalid_request")
        if len(set(v["prepared"]["bundle_id"] for v in variants)) != len(variants):
            raise LedgerError("invalid_request")
        slots = plan["replicas"]
        if (len({(s["variant_id"], s["replica_index"]) for s in slots}) != len(slots)
                or len({s["session_request_id"] for s in slots}) != len(slots)
                or len({s["protocol_id"] for s in slots}) != len(slots)):
            raise LedgerError("invalid_request")
        by_variant = defaultdict(list)
        for slot in slots:
            if slot["variant_id"] not in variant_ids:
                raise LedgerError("invalid_request")
            by_variant[slot["variant_id"]].append(slot)
        if any(not 1 <= len(by_variant[v]) <= 8 or len({s["seed"] for s in by_variant[v]})
               != len(by_variant[v]) for v in variant_ids):
            raise LedgerError("invalid_request")
        definition = plan["analysis_definition"]
        actual_rows, proofs = [], []
        for variant in variants:
            prepared = variant["prepared"]
            if variant["binding_sha256"] != content_hash({
                    "prepared": prepared, "template_sha256": plan["template_sha256"]}):
                raise LedgerError("evidence_mismatch")
            async with self.ledger.pool.acquire() as conn:
                await self.inputs._prepared_row(conn, scope_id, prepared)
                operation = await conn.fetchrow('''SELECT state,receipt FROM neomd_service.preparation_operations
                    WHERE scope_id=$1 AND operation_id=$2 AND kind='component_assemble' ''',
                    _uuid(scope_id), _uuid(variant["provenance"]["assembly_operation_id"]))
            if operation is None or operation["state"] != "succeeded" or operation["receipt"] is None:
                raise LedgerError("evidence_mismatch")
            job = {"prepared_directory": str(self.inputs._bundle_directory(prepared["bundle_id"])),
                   "prepared": prepared, "assembly_receipt": operation["receipt"],
                   "definition": definition}
            deadline = time.monotonic() + self.inputs.verification_timeout_seconds
            async with self.inputs._verification_slot(job, deadline):
                proof = await self.inputs._run_verifier(
                    job, deadline, target=_plan_provenance_process, result_max_bytes=65536)
            if set(proof) != {"actual"}:
                raise LedgerError(proof.get("error", "evidence_mismatch"))
            actual = proof["actual"]
            actual_rows.append({"variant_id": variant["variant_id"], "prepared": prepared,
                                "binding_sha256": variant["binding_sha256"],
                                "assembly_operation_id": variant["provenance"]["assembly_operation_id"],
                                "actual": actual})
            for slot in by_variant[variant["variant_id"]]:
                async with self.ledger.pool.acquire() as conn:
                    registered = await conn.fetchrow('''SELECT binding,descriptor,method,content_sha256
                        FROM neomd_service.protocol_assets WHERE scope_id=$1 AND protocol_id=$2''',
                        _uuid(scope_id), _uuid(slot["protocol_id"]))
                if registered is None:
                    raise LedgerError("evidence_mismatch")
                binding = await self.inputs.bind_protocol_template(scope_id, {
                    "prepared": prepared, "template": plan["template"],
                    "reference_bindings": variant["reference_bindings"],
                    "replica_seed": slot["seed"],
                })
                _check_slot_binding(slot, prepared, plan["template_sha256"], registered, binding)
                proofs.append({key: slot[key] for key in (
                    "variant_id", "replica_index", "seed", "protocol_id", "protocol_sha256")}
                    | {"template_sha256": plan["template_sha256"]})
        controls = _compare_controls(variants, actual_rows)
        receipt = {"schema_version": 1, "plan_sha256": content_hash(plan),
                   "template_sha256": plan["template_sha256"],
                   "analysis_definition_sha256": content_hash(definition),
                   "variants": actual_rows, "controls": controls, "slots": proofs,
                   "confounders": [c["field"] for c in controls if c["status"] == "differs"]}
        validate(f"{SCHEMA}#/$defs/PlanAttestation", receipt)
        return receipt

    async def initialize(self):
        """A restart never reruns an uncertain scientific analysis."""
        async with self.ledger.pool.acquire() as conn:
            await conn.execute('''UPDATE neomd_service.comparison_analyses
                SET state='interrupted',error='{"code":"worker_lost","message":"analysis host restarted"}'::jsonb
                WHERE state IN ('queued','running')''')

    async def close(self):
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    @staticmethod
    def _operation(row):
        return {"schema_version": 1, "operation_id": str(row["operation_id"]),
                "request_id": str(row["request_id"]), "request_sha256": row["request_sha256"],
                "comparison_id": str(row["comparison_id"]), "state": row["state"],
                "receipt": row["receipt"], "error": row["error"]}

    async def analysis_by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as conn:
            row = await conn.fetchrow('''SELECT * FROM neomd_service.comparison_analyses
                WHERE scope_id=$1 AND request_id=$2''', _uuid(scope_id), _uuid(request_id))
        if row is None:
            raise LedgerError("not_found")
        return self._operation(row)

    async def analysis(self, scope_id, operation_id):
        async with self.ledger.pool.acquire() as conn:
            row = await conn.fetchrow('''SELECT * FROM neomd_service.comparison_analyses
                WHERE scope_id=$1 AND operation_id=$2''', _uuid(scope_id), _uuid(operation_id))
        if row is None:
            raise LedgerError("not_found")
        return self._operation(row)

    async def submit_analysis(self, scope_id, request):
        validate(f"{SCHEMA}#/$defs/AnalysisRequest", request)
        if (content_hash(request["definition"]) != request["definition_sha256"]
                or len({(m["variant_id"], m["replica_index"]) for m in request["members"]})
                != len(request["members"])):
            raise LedgerError("invalid_request")
        digest = content_hash(request)
        async with self.ledger.pool.acquire() as conn, conn.transaction():
            operation_id = uuid4()
            row = await conn.fetchrow('''INSERT INTO neomd_service.comparison_analyses
                (operation_id,scope_id,request_id,comparison_id,request_sha256,request,state)
                VALUES($1,$2,$3,$4,$5,$6,'queued')
                ON CONFLICT(scope_id,request_id) DO NOTHING RETURNING *''',
                operation_id, _uuid(scope_id), _uuid(request["request_id"]),
                _uuid(request["comparison_id"]), digest, request)
            if row is None:
                row = await conn.fetchrow('''SELECT * FROM neomd_service.comparison_analyses
                    WHERE scope_id=$1 AND request_id=$2 FOR UPDATE''',
                    _uuid(scope_id), _uuid(request["request_id"]))
                if row["request_sha256"] != digest or row["request"] != request:
                    raise LedgerError("idempotency_conflict")
                return self._operation(row)
        task = asyncio.create_task(self._execute(scope_id, str(operation_id), request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return self._operation(row)

    async def _execute(self, scope_id, operation_id, request):
        async with self.ledger.pool.acquire() as conn:
            await conn.execute('''UPDATE neomd_service.comparison_analyses SET state='running'
                WHERE operation_id=$1 AND scope_id=$2 AND state='queued' ''',
                _uuid(operation_id), _uuid(scope_id))
        try:
            jobs = await self._source_jobs(scope_id, request)
            job = {"request": request, "jobs": jobs}
            deadline = time.monotonic() + min(120, self.inputs.verification_timeout_seconds)
            async with self.inputs._verification_slot(job, deadline):
                output = await self.inputs._run_verifier(
                    job, deadline, target=_analysis_process, result_max_bytes=8_388_608)
            if set(output) != {"receipt"}:
                raise LedgerError(output.get("error", "integrity_mismatch"))
            receipt = output["receipt"]
            validate(f"{SCHEMA}#/$defs/AnalysisReceipt", receipt)
            if (receipt["request_sha256"] != content_hash(request)
                    or receipt["comparison_id"] != request["comparison_id"]):
                raise LedgerError("integrity_mismatch")
            state, error = "succeeded", None
        except asyncio.CancelledError:
            async with self.ledger.pool.acquire() as conn:
                await conn.execute('''UPDATE neomd_service.comparison_analyses
                    SET state='interrupted',error='{"code":"worker_lost","message":"analysis host stopped"}'::jsonb
                    WHERE operation_id=$1 AND state IN ('queued','running')''', _uuid(operation_id))
            return
        except LedgerError as failure:
            state, error, receipt = "failed", {"code": failure.code, "message": failure.code}, None
        except Exception:
            state, error, receipt = "failed", {"code": "integrity_mismatch",
                                                 "message": "analysis source verification failed"}, None
        async with self.ledger.pool.acquire() as conn:
            await conn.execute('''UPDATE neomd_service.comparison_analyses
                SET state=$2,receipt=$3,error=$4 WHERE operation_id=$1 AND state='running' ''',
                _uuid(operation_id), state, receipt, error)

    async def _source_jobs(self, scope_id, request):
        jobs = []
        source_bytes = 0
        for member in request["members"]:
            job = {"reason": "unsubmitted", "intervention_state": "unknown"}
            if member["source"] is not None:
                source_bytes += sum(row["size_bytes"] for row in member["source"].values())
                if source_bytes > 256_000_000:
                    raise LedgerError("payload_too_large")
            if member["session_id"] is None:
                if member["scientific_state"] != "unsubmitted" or member["source"] is not None:
                    raise LedgerError("evidence_mismatch")
                jobs.append(job)
                continue
            async with self.ledger.pool.acquire() as conn:
                session = await conn.fetchrow('''SELECT s.snapshot,s.request_id,p.binding,p.descriptor,
                    p.content_sha256,p.protocol_id FROM neomd_service.sessions s
                    JOIN neomd_service.protocol_assets p ON p.scope_id=s.scope_id
                    AND p.protocol_id=s.protocol_id WHERE s.scope_id=$1 AND s.session_id=$2''',
                    _uuid(scope_id), _uuid(member["session_id"]))
                if session is None:
                    raise LedgerError("not_found")
                if (str(session["request_id"]) != member["session_request_id"]
                        or str(session["protocol_id"]) != member["protocol_id"]
                        or session["binding"] != member["prepared"]
                        or session["content_sha256"] != member["protocol_sha256"]
                        or session["descriptor"]["method"] != "prod"
                        or session["descriptor"]["physics"]["seed"] != member["seed"]):
                    raise LedgerError("evidence_mismatch")
                snapshot = session["snapshot"]
                if snapshot["state"] != member["scientific_state"]:
                    raise LedgerError("evidence_mismatch")
                commands = await conn.fetch('''SELECT command_id,request,receipt,status FROM neomd_service.commands
                    WHERE scope_id=$1 AND session_id=$2 ORDER BY command_id''',
                    _uuid(scope_id), _uuid(member["session_id"]))
                applied = [{"command_id": str(c["command_id"]), "request": c["request"],
                            "receipt": c["receipt"]} for c in commands if c["status"] == "applied"]
                digest = content_hash(applied)
                if member["applied_command_sha256"] is not None and member["applied_command_sha256"] != digest:
                    raise LedgerError("evidence_mismatch")
                perturbed = any(c["request"]["payload"]["type"] in {
                    "set_forces", "set_restraint", "clear_restraint", "set_parameter"}
                    for c in commands if c["status"] == "applied")
                uncertain = any(c["status"] in {"claimed", "indeterminate"} for c in commands)
                job["intervention_state"] = "unknown" if uncertain else (
                    "perturbed" if perturbed else "unperturbed")
                if snapshot["state"] != "completed":
                    job["reason"] = "not_completed"
                elif job["intervention_state"] != "unperturbed":
                    job["reason"] = "perturbed"
                elif member["source"] is None:
                    job["reason"] = ("unpublished" if member["publication_state"] != "complete"
                                     else "source_unavailable")
                else:
                    final = snapshot.get("finalization")
                    if final is None or final["status"] != "complete":
                        raise LedgerError("artifact_unverified")
                    paths = {}
                    for kind, expected_name in (("result_manifest", "result-manifest.json"),
                                                ("trajectory", "output.dcd"),
                                                ("state", "output.state")):
                        source = member["source"][kind]
                        row = await conn.fetchrow('''SELECT descriptor FROM neomd_service.artifacts
                            WHERE scope_id=$1 AND session_id=$2 AND artifact_id=$3''',
                            _uuid(scope_id), _uuid(member["session_id"]),
                            _uuid(source["artifact_id"]))
                        if (row is None or source["artifact_id"] not in final["artifact_ids"]
                                or row["descriptor"]["original_filename"] != expected_name
                                or row["descriptor"]["content_sha256"] != source["sha256"]
                                or row["descriptor"]["size_bytes"] != source["size_bytes"]):
                            raise LedgerError("artifact_unverified")
                        # Publication may retire the object link before a
                        # comparison is requested. The producer's sealed output
                        # remains under this exact session attempt; the verifier
                        # rechecks its size, SHA-256 and file identity.
                        paths[kind] = str(self.inputs.root / "sessions"
                                          / str(_uuid(member["session_id"]))
                                          / str(_uuid(snapshot["attempt_id"]))
                                          / "output" / expected_name)
                    assembly = await conn.fetchrow('''SELECT state,receipt FROM neomd_service.preparation_operations
                        WHERE scope_id=$1 AND operation_id=$2 AND kind='component_assemble' ''',
                        _uuid(scope_id), _uuid(member["assembly_operation_id"]))
                    if (assembly is None or assembly["state"] != "succeeded"
                            or assembly["receipt"]["result"]["sha256"]
                            != member["prepared"]["manifest_sha256"]):
                        raise LedgerError("evidence_mismatch")
                    job.update(reason="complete", paths=paths,
                               prepared_directory=str(self.inputs._bundle_directory(
                                   member["prepared"]["bundle_id"])),
                               assembly_receipt=assembly["receipt"])
            jobs.append(job)
        return jobs


def _analysis_process(job, _limits, result, parent_pid):
    from .comparison_analysis import analyse_cohort

    arm_parent_death(parent_pid)
    try:
        outcome = {"receipt": analyse_cohort(job["request"], job["jobs"])}
    except Exception:
        outcome = {"error": "integrity_mismatch"}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()


def _plan_provenance_process(job, _limits, result, parent_pid):
    arm_parent_death(parent_pid)
    try:
        actual, manifest, mapping, assembly = _provenance(
            job["prepared_directory"], job["prepared"], job["assembly_receipt"])
        _selection(job["definition"], manifest, mapping, assembly)
        outcome = {"actual": actual}
    except Exception:
        outcome = {"error": "evidence_mismatch"}
    result.put(outcome, timeout=1)
    result.flush_until(time.monotonic() + 1)
    result.close()
