"""Durable protein preparation: staged sources, one isolated executor, verified receipts.

The lifecycle runner owns execution; HTTP only commits intent or reads receipts.
Scientific work happens in spawned, parent-death-armed children with explicit
deadlines and address-space limits. Restart reconciliation interrupts every
non-terminal operation and never restarts a scientific task.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
import queue
import secrets
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from .contract import canonical_bytes, content_hash, validate
from .durability import sync_directories
from .ipc import Channel
from .ledger import LedgerError
from .preparation import preparation_catalog
from .preparation_execution import (
    SCHEMA,
    _prepare_process,
    _prepare_topology_process,
    _validate_source_process,
    _validate_topology_source_process,
    _verify_process,
    _verify_topology_process,
)

CATALOG = "urn:neodyn:protein-preparation:v1:catalog"
RECEIPT = "urn:neodyn:protein-preparation:v1:receipt"
INVENTORY = "urn:neodyn:protein-preparation:v1:inventory"
QC = "urn:neodyn:protein-preparation:v1:qc"
BEGIN = f"{SCHEMA}#/$defs/SourceUploadBeginRequest"
UPLOAD_RECEIPT = f"{SCHEMA}#/$defs/SourceUploadReceipt"
PREPARATION_REQUEST = f"{SCHEMA}#/$defs/PreparationRequest"
TOPOLOGY_SCHEMA = "urn:neodyn:protein-preparation:v2:operation"
TOPOLOGY_CATALOG = "urn:neodyn:protein-preparation:v2:catalog"
TOPOLOGY_BEGIN = f"{TOPOLOGY_SCHEMA}#/$defs/SourceUploadBeginRequest"
TOPOLOGY_UPLOAD_RECEIPT = f"{TOPOLOGY_SCHEMA}#/$defs/SourceUploadReceipt"
TOPOLOGY_REQUEST = "urn:neodyn:protein-preparation:v2:request"
COMPONENT_SCHEMA = "urn:neodyn:component-operations:v1:operation"
COMPONENT_BLOB = COMPONENT_SCHEMA
COMPONENT_REQUEST = f"{COMPONENT_SCHEMA}#/$defs/ComponentRequest"
COMPONENT_KINDS = frozenset({"component_review", "component_generate",
                             "component_import", "component_assemble"})
KIND = "protein_preparation"
TOPOLOGY_KIND = "topology_preparation"
PENDING = {"queued", "running"}
SYSTEM_PARAMETERS = {
    "hydrogen_mass_amu": 4.0, "constraints": "HBonds", "rigid_water": True,
    "remove_cm_motion": False, "nonbonded_method": "pme", "nonbonded_cutoff_nm": 1.0,
    "ewald_error_tolerance": 0.0005, "use_dispersion_correction": True,
    "switch_distance_nm": None, "exceptions_use_periodic_boundary_conditions": False,
    "flexible_constraints": False,
}
ROLES = (
    ("manifest.json", "manifest"), ("topology.pdbx", "topology"), ("system.xml", "system"),
    ("coordinates.f64le", "coordinates"), ("mapping.json", "mapping"),
    ("assembly-report.json", "report"), ("preparation.log", "log"),
    ("source-topology-map.json", "mapping"), ("topology-changes.json", "mapping"),
    ("ligand.json", "ligand"),
)


def _stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _uuid(value):
    return UUID(str(value))


def _role(path):
    for name, role in ROLES:
        if path == name:
            return role
    if path.startswith("sources/"):
        return "source"
    if path.startswith("forcefield/"):
        return "forcefield"
    raise LedgerError("integrity_mismatch")


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _gaff_provider_ready():
    try:
        from importlib.resources import files

        from openmmforcefields.generators import GAFFTemplateGenerator

        from neomd.component_gaff import _GAFF, _ambertools_receipt, _BoundedRunner
        from neomd.tools.antechamber import AntechamberBackend

        if _GAFF not in GAFFTemplateGenerator.INSTALLED_FORCEFIELDS:
            return False
        runner = _BoundedRunner()
        _, resources = _ambertools_receipt(runner)
        gaff_dat = Path(AntechamberBackend(runner, gaff_version=_GAFF).gaff_dat_filename)
        resource_paths = (
            (f"gaff/{_GAFF}.xml", gaff_dat.parent.parent / "ffxml" / f"{_GAFF}.xml"),
            (f"gaff/{_GAFF}.dat", gaff_dat),
            ("amber/protein.ff14SB.xml", files("openmmforcefields") / "ffxml" / "amber" / "protein.ff14SB.xml"),
            ("amber/tip3p_standard.xml", files("openmmforcefields") / "ffxml" / "amber" / "tip3p_standard.xml"),
        )
        resources.extend({"resource_id": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in resource_paths)
        return content_hash(resources) == _COMPONENT_SCOPES["gaff2"]["resource_sha256"]
    except (ImportError, KeyError, OSError, ValueError):
        return False


def _openff_provider_ready():
    from neomd.component_openff import openff_provider_ready

    return openff_provider_ready()


def _sep_provider_ready():
    from neomd.component_sep import sep_provider_ready

    return sep_provider_ready()


# Canonical P10-sage-qualification.json: exact JZ4 package, same-method
# reference, two-copy System QC and prepared/v2 closure. Runtime resource
# readiness still gates the catalog and every new Generate admission.
_OPENFF_QUALIFICATION_SHA256 = "edcdb71df11b9f4e670d545d33e32cf7b8d2aa896cffe1df0c47f07fd6efbe01"
# Canonical task qualification binds the exact curated 2MZ2 source/review,
# independent LEaP junction comparison, full-chain package and prepared QC.
_SEP_QUALIFICATION_SHA256 = "21f097a705f94f5d64105a6794c5590e206294f9cf300eca11d6bcc53fc0811b"
_IMPORTED_GAFF_FORCE_SHA256 = "2b29b08cf94f5f7cb86ade4503c1b3ebb0224471169df5cfa4bae8d88806eac7"

# Bounded CPU evidence, not a promise that every state in a chemistry family
# has been qualified. Resource digests are RFC 8785 hashes of the retained
# resource sets; import's set contains the exact supplied JZ4 force asset.
_COMPONENT_SCOPES = {
    "stored-bond-v1": {
        "chemistry_scope": "repaired 3HTB canonical protein with source-bound stored bonds",
        "states": [{"state_sha256": "a612997d71dd309428b2739a799478ae040e76b73a1b929e49f089fc5fc67d8e",
                    "formal_charge": 6}],
        "platform": "CPU",
        "resource_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "receipt_sha256": "93352c28d24f4354dce14f20464c26a81d9a73f6276fdd7f49c8d3f7bbf278ca",
    },
    "gaff2": {
        "chemistry_scope": "accepted achiral neutral JZ4; GAFF2/ABCG2; stereogenic TCH unqualified",
        "states": [
            {"state_sha256": "4c9c7a6da599b800021be014e6d08c07296b16ad36584570722576dec6a85ead",
             "formal_charge": 0},
            {"state_sha256": "8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e",
             "formal_charge": 0},
        ],
        "platform": "CPU",
        "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436",
        "receipt_sha256": "88a3b6b5dbf7c678638a02ed44a17b0c728ae233f0c8a5684f3957a8dd86b69e",
    },
    "openff-sage-2.2.1": {
        "chemistry_scope": "accepted achiral neutral 22-atom JZ4; fixed two-copy ff14SB/TIP3P/PME/HBonds/4-amu profile",
        "states": [{"state_sha256": "8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e",
                    "formal_charge": 0}],
        "platform": "CPU",
        "resource_sha256": "1ac02bcfe320ad06b9c6a89ccc2c60c0303368e5e6ad893a4b841baf3dbffc4c",
        "receipt_sha256": "edcdb71df11b9f4e670d545d33e32cf7b8d2aa896cffe1df0c47f07fd6efbe01",
    },
    "amber-ff14sb-phosaa14sb-sep": {
        "chemistry_scope": "one complete explicit-H 2MZ2 chain with one internal SEP −2; no mixed package family",
        "states": [{"state_sha256": "e6ff85882095df58577cefad1d9b65836a09cfbe18f4ade5929065f08da1c103",
                    "formal_charge": -2}],
        "platform": "CPU",
        "resource_sha256": "1e80f0dde9b6c5ded8cc65cf5c84bfc765d556915ad1aca5ed40634a4c405a98",
        "receipt_sha256": _SEP_QUALIFICATION_SHA256,
    },
    "declarative": {
        "chemistry_scope": "accepted neutral 22-atom JZ4 declarative import retaining exact supplied GAFF force bytes",
        "states": [
            {"state_sha256": "4c9c7a6da599b800021be014e6d08c07296b16ad36584570722576dec6a85ead",
             "formal_charge": 0},
            {"state_sha256": "8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e",
             "formal_charge": 0},
        ],
        "platform": "CPU",
        "resource_sha256": "cd0dbddb5f1c1efa0109636b1ddf08f0362ec490afa4d163e1835b18793ec397",
        "receipt_sha256": "92c3f68d3a2c3d1830745ab36b5ce12fd4a2fbe1a738552afefeba22fac7688c",
    },
    "stored-package-v1": {
        "chemistry_scope": "source-complete two-JZ4 or one-internal-SEP 2MZ2 profiles",
        "states": [
            {"state_sha256": "4c9c7a6da599b800021be014e6d08c07296b16ad36584570722576dec6a85ead",
             "formal_charge": 0},
            {"state_sha256": "8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e",
             "formal_charge": 0},
            {"state_sha256": "e6ff85882095df58577cefad1d9b65836a09cfbe18f4ade5929065f08da1c103",
             "formal_charge": -2},
        ],
        "profiles": [
            {"source_sha256": "715cfce1bac3d16c8a799197ff936ba52ee31718aff8e6a530574a44fd2f8a55",
             "state_sha256s": ["4c9c7a6da599b800021be014e6d08c07296b16ad36584570722576dec6a85ead"],
             "package_methods": [
                 {"method_id": "amber.gaff2.abcg2", "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436"}],
             "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436",
             "receipt_sha256": "3541028731140f1a6d3597a9b2534ceb791553d563540bd22e91874ee72e708a"},
            {"source_sha256": "80ecfeeef28fa03804726c6327e5edd4abd810b5e9a39f12d4851dd33203cd46",
             "state_sha256s": ["8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e"],
             "package_methods": [
                 {"method_id": "amber.gaff2.abcg2", "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436"}],
             "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436",
             "receipt_sha256": "88a3b6b5dbf7c678638a02ed44a17b0c728ae233f0c8a5684f3957a8dd86b69e"},
            {"source_sha256": "80ecfeeef28fa03804726c6327e5edd4abd810b5e9a39f12d4851dd33203cd46",
             "state_sha256s": ["8fb219dbe3e00fcb568818f65f67e60b4fa51a6223199994ea4290c1d16cc39e"],
             "package_methods": [
                 {"method_id": "openff.sage2.2.1.am1bcc", "resource_sha256": "957c5b5a3ecdb0544005ea7c82bd2878e418cecea9215d1ff1452dd43e14e422"}],
             "resource_sha256": "1ac02bcfe320ad06b9c6a89ccc2c60c0303368e5e6ad893a4b841baf3dbffc4c",
             "receipt_sha256": _OPENFF_QUALIFICATION_SHA256},
            {"source_sha256": "0a78ef241d94f06f34013df2252278da98f54dca0d71035c9bf8758d067557a0",
             "state_sha256s": ["e6ff85882095df58577cefad1d9b65836a09cfbe18f4ade5929065f08da1c103"],
             "package_methods": [
                 {"method_id": "openmm.ff14sb.phosaa14sb.sep", "resource_sha256": "9d281b9dbebe635e12d5071be28b21aac418cb0e1f165458000e4315bfeb596b"}],
             "resource_sha256": "1e80f0dde9b6c5ded8cc65cf5c84bfc765d556915ad1aca5ed40634a4c405a98",
             "receipt_sha256": _SEP_QUALIFICATION_SHA256},
        ],
        "platform": "CPU",
        "resource_sha256": "ccda88dc7ab4e821606ec3cbb148e1cfab72c20a5a11e0baa77ad5617980c436",
        "receipt_sha256": "3541028731140f1a6d3597a9b2534ceb791553d563540bd22e91874ee72e708a",
    },
}


@dataclass(frozen=True)
class PreparationLimits:
    max_source_bytes: int
    max_result_bytes: int
    time_limit_seconds: int
    memory_limit_bytes: int
    max_queued_per_owner: int
    max_concurrent_operations: int
    max_request_bytes: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**53 - 1:
                raise ValueError(f"{name} must be a positive safe integer")
        if self.max_concurrent_operations != 1:
            raise ValueError("the isolated preparation executor runs one operation at a time")


class PreparationStore:
    """Host-owned layout; remote operations address only UUIDs and exact hashes."""

    def __init__(self, ledger, root, *, pool_id, capacity_bytes, limits: PreparationLimits,
                 shutdown_timeout_seconds: int):
        self.ledger = ledger
        self.root = Path(root).resolve()
        self.pool_id = pool_id
        if type(capacity_bytes) is not int or capacity_bytes < 1:
            raise ValueError("preparation capacity must be a positive integer")
        self.capacity_bytes = capacity_bytes
        self.limits = limits
        if type(shutdown_timeout_seconds) is not int or shutdown_timeout_seconds < 1:
            raise ValueError("preparation shutdown timeout must be a positive integer")
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.owner_id = uuid4()
        self.signal = asyncio.Event()
        self.task = None
        self.closing = False
        self.faulted = False
        self._children = set()
        self._submission_slot = asyncio.Lock()
        self._initialized = False
        catalog = preparation_catalog()
        self._preset = catalog["presets"][0]
        from .preparation import topology_preparation_catalog

        self._topology_preset = topology_preparation_catalog()["presets"][0]

    @staticmethod
    def check_catalog_limits(limits: PreparationLimits):
        """Fail closed at startup when host settings diverge from the advertised catalog."""
        catalog = preparation_catalog()
        for name in ("max_source_bytes", "max_result_bytes", "time_limit_seconds",
                     "memory_limit_bytes", "max_queued_per_owner", "max_concurrent_operations"):
            if catalog["limits"][name] != getattr(limits, name):
                raise ValueError(f"preparation setting {name} differs from the advertised catalog limit")

    # --- lifecycle ---------------------------------------------------------

    async def initialize(self, *, reconcile=True, start=True):
        if self.faulted:
            raise LedgerError("service_unavailable")
        if self._initialized:
            return
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """INSERT INTO neomd_service.storage_capacity VALUES($1,$2)
                ON CONFLICT(pool_id) DO NOTHING""",
                self.pool_id, self.capacity_bytes,
            )
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1",
                self.pool_id,
            )
            if capacity != self.capacity_bytes:
                raise LedgerError("service_unavailable")
            await connection.execute(
                """INSERT INTO neomd_service.preparation_policy VALUES($1,$2)
                ON CONFLICT(pool_id) DO UPDATE SET max_queued_per_owner=excluded.max_queued_per_owner""",
                self.pool_id, self.limits.max_queued_per_owner,
            )
        self._initialized = True
        if reconcile:
            await self._reconcile()
        if start:
            self.task = asyncio.create_task(self._run())

    async def _reconcile(self):
        """Restart fences every non-terminal operation; nothing scientific restarts."""
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                """SELECT * FROM neomd_service.preparation_operations
                WHERE pool_id=$1 AND state IN ('queued','running') FOR UPDATE""",
                self.pool_id,
            )
            for row in rows:
                await self._write(connection, row, "interrupted", code="worker_lost",
                                  message="Service restart fenced the non-terminal operation; "
                                          "retry requires a new request_id.")
        # Interrupted running writers retain their reservation and files. A
        # restart/lease fence is not proof of their exit or storage reclamation.

    async def close(self):
        self.closing = True
        self.signal.set()
        if self.task is not None:
            self.task.cancel()
            done, _ = await asyncio.wait({self.task}, timeout=self.shutdown_timeout_seconds)
            if not done:
                self.faulted = True
            elif not self.task.cancelled():
                self.task.exception()
            self.task = None
        for process in tuple(self._children):
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)
            process.close()
            self._children.discard(process)
        self._initialized = False

    # --- catalog -----------------------------------------------------------

    async def catalog(self, scope_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        catalog = preparation_catalog()
        if self.closing or not self._initialized:
            catalog["unavailable"] = {
                "reason": "executor_unavailable",
                "detail": "The preparation executor is not running on this service instance.",
            }
        validate(CATALOG, catalog)
        return catalog

    async def topology_catalog(self, scope_id):
        from .preparation import topology_preparation_catalog

        if self.faulted:
            raise LedgerError("service_unavailable")
        catalog = topology_preparation_catalog()
        if self.closing or not self._initialized:
            catalog["unavailable"] = {
                "reason": "executor_unavailable",
                "detail": "The preparation executor is not running on this service instance.",
            }
        validate(TOPOLOGY_CATALOG, catalog)
        return catalog

    async def component_catalog(self, scope_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        available = self._initialized and not self.closing
        gaff_ready = available and await asyncio.to_thread(_gaff_provider_ready)
        openff_ready = (available and _OPENFF_QUALIFICATION_SHA256 is not None
                        and await asyncio.to_thread(_openff_provider_ready))
        sep_ready = (available and _SEP_QUALIFICATION_SHA256 is not None
                     and await asyncio.to_thread(_sep_provider_ready))
        methods = [
            ("component_review", "stored-bond-v1",
             "93352c28d24f4354dce14f20464c26a81d9a73f6276fdd7f49c8d3f7bbf278ca",
             available),
            ("component_generate", "gaff2",
             "3821861ceb115f389aa2860b28c5c16dd86856c5816626b117a309dbe6aef847",
             gaff_ready),
            ("component_generate", "openff-sage-2.2.1",
             _OPENFF_QUALIFICATION_SHA256, openff_ready),
            ("component_generate", "amber-ff14sb-phosaa14sb-sep",
             _SEP_QUALIFICATION_SHA256, sep_ready),
            ("component_import", "declarative",
             "92c3f68d3a2c3d1830745ab36b5ce12fd4a2fbe1a738552afefeba22fac7688c",
             available),
            ("component_assemble", "stored-package-v1",
             "3541028731140f1a6d3597a9b2534ceb791553d563540bd22e91874ee72e708a",
             available),
        ]
        catalog = {
            "schema_version": 1,
            "methods": [{"kind": kind, "method_id": method,
                         "availability": "available" if ready else "unavailable",
                         **({"qualification_sha256": sha,
                             "scope": copy.deepcopy(_COMPONENT_SCOPES[method])} if ready else {})}
                        for kind, method, sha, ready in methods] +
                       [{"kind": "component_generate", "method_id": "amber-ff14sb",
                         "availability": "unavailable"}],
            "limits": {"max_input_count": 64,
                       "max_blob_bytes": self.limits.max_source_bytes,
                       "max_result_bytes": self.limits.max_result_bytes},
        }
        validate("urn:neodyn:component-operations:v1:catalog", catalog)
        return catalog

    # --- source upload lane --------------------------------------------------

    def _upload_directory(self, upload_id):
        return self.root / "preparation-uploads" / str(_uuid(upload_id))

    async def _used_capacity(self, connection):
        return await connection.fetchval(
            """SELECT (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.asset_uploads WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.output_reservations WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_bytes),0) FROM neomd_service.preparation_source_uploads WHERE pool_id=$1)
            + (SELECT coalesce(sum(reserved_output_bytes),0) FROM neomd_service.preparation_operations WHERE pool_id=$1)
            + (SELECT coalesce(sum(pa.size_bytes),0) FROM neomd_service.preparation_artifacts pa
               JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id
               WHERE po.pool_id=$1)""",
            self.pool_id,
        )

    async def begin_source(self, scope_id, request):
        topology = request.get("schema_version") == 2
        component = request.get("schema_version") == 3
        validate(COMPONENT_BLOB + "#/$defs/BlobUploadBeginRequest" if component else
                 TOPOLOGY_BEGIN if topology else BEGIN, request)
        if self.faulted or not self._initialized:
            raise LedgerError("service_unavailable")
        if request["size_bytes"] > self.limits.max_source_bytes:
            raise LedgerError("payload_too_large")
        digest = content_hash({key: value for key, value in request.items() if key != "request_id"})
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                """SELECT content_sha256,receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND request_id=$2""",
                _uuid(scope_id), _uuid(request["request_id"]),
            )
            if existing is not None:
                if existing["content_sha256"] != digest:
                    raise LedgerError("idempotency_conflict")
                return existing["receipt"]
            used = await self._used_capacity(connection)
            if request["size_bytes"] > capacity - used:
                raise LedgerError("admission_full")
            now = await connection.fetchval("SELECT clock_timestamp()")
            receipt = {
                "schema_version": 3 if component else 2 if topology else 1,
                "upload_id": str(uuid4()),
                "request_id": request["request_id"],
                "format": request["format"],
                "size_bytes": request["size_bytes"],
                "state": "registered",
                "created_at": _stamp(now),
            }
            if component:
                receipt["content_sha256"] = request["content_sha256"]
            elif topology:
                receipt.update({key: request[key] for key in (
                    "schema_id", "content_sha256", "decoded_size_bytes", "topology_revision_id")})
            else:
                receipt["canonical_bytes_sha256"] = request["canonical_bytes_sha256"]
            validate(COMPONENT_BLOB + "#/$defs/BlobUploadReceipt" if component else
                     TOPOLOGY_UPLOAD_RECEIPT if topology else UPLOAD_RECEIPT, receipt)
            await connection.execute(
                """INSERT INTO neomd_service.preparation_source_uploads
                (upload_id,scope_id,request_id,pool_id,content_sha256,request,receipt,reserved_bytes)
                VALUES($1,$2,$3,$4,$5,$6,$7,$8)""",
                _uuid(receipt["upload_id"]), _uuid(scope_id), _uuid(request["request_id"]),
                self.pool_id, digest, request, receipt, request["size_bytes"],
            )
            return receipt

    async def source_upload(self, scope_id, upload_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.preparation_source_uploads WHERE scope_id=$1 AND upload_id=$2",
                _uuid(scope_id), _uuid(upload_id),
            )
        if receipt is None:
            raise LedgerError("not_found")
        return receipt

    async def topology_source_upload(self, scope_id, upload_id):
        receipt = await self.source_upload(scope_id, upload_id)
        if receipt["schema_version"] != 2:
            raise LedgerError("not_found")
        return receipt

    async def protein_source_upload(self, scope_id, upload_id):
        receipt = await self.source_upload(scope_id, upload_id)
        if receipt["schema_version"] != 1:
            raise LedgerError("not_found")
        return receipt

    async def begin_component_blob(self, scope_id, request):
        if request.get("schema_version") != 3:
            raise LedgerError("invalid_request")
        return await self.begin_source(scope_id, request)

    async def component_blob_upload(self, scope_id, upload_id):
        receipt = await self.source_upload(scope_id, upload_id)
        if receipt["schema_version"] != 3:
            raise LedgerError("not_found")
        return receipt

    async def component_blob_by_request(self, scope_id, request_id):
        receipt = await self.source_upload_by_request(scope_id, request_id)
        if receipt["schema_version"] != 3:
            raise LedgerError("not_found")
        return receipt

    async def source_upload_by_request(self, scope_id, request_id):
        async with self.ledger.pool.acquire() as connection:
            receipt = await connection.fetchval(
                "SELECT receipt FROM neomd_service.preparation_source_uploads WHERE scope_id=$1 AND request_id=$2",
                _uuid(scope_id), _uuid(request_id),
            )
        if receipt is None:
            raise LedgerError("not_found")
        return receipt

    async def topology_source_upload_by_request(self, scope_id, request_id):
        receipt = await self.source_upload_by_request(scope_id, request_id)
        if receipt["schema_version"] != 2:
            raise LedgerError("not_found")
        return receipt

    async def protein_source_upload_by_request(self, scope_id, request_id):
        receipt = await self.source_upload_by_request(scope_id, request_id)
        if receipt["schema_version"] != 1:
            raise LedgerError("not_found")
        return receipt

    async def put_source_content(self, scope_id, upload_id, chunks):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """SELECT receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND upload_id=$2 FOR UPDATE""",
                _uuid(scope_id), _uuid(upload_id),
            )
            if row is None:
                raise LedgerError("not_found")
            receipt = row["receipt"]
            if receipt["state"] == "stored":
                return  # Idempotent redelivery of the exact declared bytes.
            folder = self._upload_directory(upload_id)
            await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            temporary = folder / (str(uuid4()) + ".partial")
            destination = folder / f"source.{receipt['format']}"
            digest, count = hashlib.sha256(), 0
            try:
                with temporary.open("xb") as handle:
                    iterator = chunks.__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(anext(iterator), timeout=30.0)
                        except StopAsyncIteration:
                            break
                        count += len(chunk)
                        if count > receipt["size_bytes"] or count > self.limits.max_source_bytes:
                            raise LedgerError("payload_too_large")
                        digest.update(chunk)
                        await asyncio.to_thread(handle.write, chunk)
                    await asyncio.to_thread(handle.flush)
                    await asyncio.to_thread(os.fsync, handle.fileno())
                expected_sha = (receipt["content_sha256"] if receipt["schema_version"] == 3
                                else receipt["content_sha256"] if receipt["schema_version"] == 2
                                else receipt["canonical_bytes_sha256"])
                if count != receipt["size_bytes"] or digest.hexdigest() != expected_sha:
                    raise LedgerError("integrity_mismatch")
                await asyncio.to_thread(os.replace, temporary, destination)
                await asyncio.to_thread(sync_directories, destination.parent, root=self.root)
                receipt["state"] = "stored"
                validate(COMPONENT_BLOB + "#/$defs/BlobUploadReceipt" if receipt["schema_version"] == 3
                         else TOPOLOGY_UPLOAD_RECEIPT if receipt["schema_version"] == 2
                         else UPLOAD_RECEIPT, receipt)
                await connection.execute(
                    """UPDATE neomd_service.preparation_source_uploads
                    SET receipt=$2,updated_at=clock_timestamp() WHERE upload_id=$1""",
                    _uuid(upload_id), receipt,
                )
            except TimeoutError as error:
                raise LedgerError("service_unavailable") from error
            finally:
                temporary.unlink(missing_ok=True)

    # --- operation submission and queries --------------------------------------

    async def submit(self, scope_id, request):
        # Bound validation admission before retaining requests or spawning children.
        if request.get("schema_version") != 1:
            raise LedgerError("invalid_request")
        if self._submission_slot.locked():
            raise LedgerError("admission_full")
        async with self._submission_slot:
            return await self._submit(scope_id, request, validated=False)

    async def submit_topology(self, scope_id, request):
        if request.get("schema_version") != 2:
            raise LedgerError("invalid_request")
        if self._submission_slot.locked():
            raise LedgerError("admission_full")
        async with self._submission_slot:
            return await self._submit(scope_id, request, validated=False)

    async def submit_component(self, scope_id, request):
        if request.get("kind") not in COMPONENT_KINDS:
            raise LedgerError("invalid_request")
        if self._submission_slot.locked():
            raise LedgerError("admission_full")
        async with self._submission_slot:
            return await self._submit_component(scope_id, request, validated=False)

    async def _component_bindings(self, connection, scope, refs):
        bindings = []
        for ref in refs:
            binding = dict(ref, resolved_owner_scope=str(scope))
            if "upload_id" in ref:
                row = await connection.fetchrow(
                    """SELECT receipt FROM neomd_service.preparation_source_uploads
                    WHERE scope_id=$1 AND upload_id=$2""", scope, _uuid(ref["upload_id"]))
                if row is None:
                    raise LedgerError("evidence_mismatch")
                receipt = row["receipt"]
                if (receipt["schema_version"] != 3 or receipt["state"] != "stored"
                        or receipt["content_sha256"] != ref["sha256"]
                        or receipt["size_bytes"] != ref["size_bytes"]):
                    raise LedgerError("evidence_mismatch")
            else:
                row = await connection.fetchrow(
                    """SELECT po.state,po.kind,po.receipt,pa.path,pa.sha256,pa.size_bytes
                    FROM neomd_service.preparation_artifacts pa
                    JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id
                    WHERE po.scope_id=$1 AND pa.scope_id=$1 AND po.operation_id=$2 AND pa.artifact_id=$3""",
                    scope, _uuid(ref["operation_id"]), _uuid(ref["artifact_id"]))
                if (row is None or row["state"] != "succeeded" or row["receipt"] is None
                        or row["sha256"] != ref["sha256"]
                        or row["size_bytes"] != ref["size_bytes"]):
                    raise LedgerError("evidence_mismatch")
                binding["producer_receipt_sha256"] = content_hash(row["receipt"])
                binding["producer_path"] = row["path"]
                binding["producer_kind"] = row["kind"]
                binding["producer_result_sha256"] = row["receipt"].get("result", {}).get("sha256")
            bindings.append(binding)
        return bindings

    def _component_input_path(self, binding):
        if "upload_id" in binding:
            return self._upload_directory(binding["upload_id"]) / "source.component-blob"
        return self._final(binding["operation_id"]) / binding["producer_path"]

    async def _materialize_component_inputs(self, bindings, directory):
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=False)
        index = {}
        for binding in bindings:
            name = binding["name"]
            source = self._component_input_path(binding)
            target = directory / name
            await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
            try:
                await asyncio.to_thread(shutil.copyfile, source, target)
            except OSError as error:
                raise LedgerError("evidence_mismatch") from error
            size = (await asyncio.to_thread(target.lstat)).st_size
            sha = await asyncio.to_thread(_digest, target)
            if size != binding["size_bytes"] or sha != binding["sha256"]:
                raise LedgerError("evidence_mismatch")
            index[name] = {"sha256": sha, "size_bytes": size}
        index_bytes = canonical_bytes(index)
        await asyncio.to_thread((directory / "index.json").write_bytes, index_bytes)
        return hashlib.sha256(index_bytes).hexdigest()

    async def _submit_component(self, scope_id, request, *, validated):
        validate(COMPONENT_REQUEST, request)
        if self.closing or self.faulted or not self._initialized:
            raise LedgerError("service_unavailable")
        if len(canonical_bytes(request)) > self.limits.max_request_bytes:
            raise LedgerError("payload_too_large")
        if sum(ref["size_bytes"] for ref in request["inputs"]) > self.limits.max_result_bytes:
            raise LedgerError("payload_too_large")
        scope = _uuid(scope_id)
        digest = content_hash(request)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id)
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                """SELECT kind,request_sha256,snapshot FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND request_id=$2 AND kind LIKE 'component_%'""",
                scope, _uuid(request["request_id"]))
            if existing is not None:
                if existing["kind"] != request["kind"] or existing["request_sha256"] != digest:
                    raise LedgerError("request_conflict")
                return existing["snapshot"]
            if request["kind"] == "component_generate":
                method = request["options"]["method_id"]
                ready = (_gaff_provider_ready if method == "gaff2" else
                         _openff_provider_ready if method == "openff-sage-2.2.1"
                         and _OPENFF_QUALIFICATION_SHA256 is not None else
                         _sep_provider_ready if method == "amber-ff14sb-phosaa14sb-sep"
                         and _SEP_QUALIFICATION_SHA256 is not None else None)
                if ready is None or not await asyncio.to_thread(ready):
                    raise LedgerError("capability_unavailable")
            bindings = await self._component_bindings(connection, scope, request["inputs"])
            if (request["kind"] == "component_generate"
                    and request["options"]["method_id"] == "amber-ff14sb-phosaa14sb-sep"):
                named = {row["name"]: row for row in bindings}
                if (set(named) != {"state.json", "instance.json", "source.json.gz"}
                        or named["instance.json"].get("producer_kind") != "component_review"
                        or named["instance.json"].get("producer_path") != "instance.json"
                        or "upload_id" not in named["source.json.gz"]):
                    raise LedgerError("evidence_mismatch")
            if request["kind"] in {"component_generate", "component_import",
                                   "component_assemble"}:
                reviewed = [row for row in bindings if row["name"] == "state.json" or
                            row["name"].startswith("states/")]
                if (not reviewed or any(
                    row.get("producer_kind") != "component_review" or
                    row.get("producer_path") != "state.json" or
                    row.get("producer_result_sha256") != row["sha256"]
                    for row in reviewed)):
                    raise LedgerError("evidence_mismatch")
            if request["kind"] in {"component_generate", "component_import"}:
                method = (request["options"]["method_id"] if request["kind"] == "component_generate"
                          else "declarative")
                accepted = {item["state_sha256"] for item in _COMPONENT_SCOPES[method]["states"]}
                if not any(row["name"] == "state.json" and row["sha256"] in accepted
                           for row in reviewed):
                    raise LedgerError("capability_unavailable")
                if request["kind"] == "component_import":
                    forces = [row for row in bindings if row["name"] == "members/parameters.json"]
                    if len(forces) != 1 or forces[0]["sha256"] != _IMPORTED_GAFF_FORCE_SHA256:
                        raise LedgerError("capability_unavailable")
            if request["kind"] == "component_assemble":
                sources = [row["sha256"] for row in bindings if row["name"] == "source.json.gz"]
                package_methods = []
                package_sources = {}
                package_count = 0
                for row in bindings:
                    if not row["name"].startswith("packages/") or not row["name"].endswith(".json"):
                        continue
                    package_count += 1
                    if (row.get("producer_kind") not in {"component_generate", "component_import"}
                            or row.get("producer_path") != "package.json"
                            or row.get("producer_result_sha256") != row["sha256"]):
                        raise LedgerError("evidence_mismatch")
                    try:
                        data = await asyncio.to_thread(self._component_input_path(row).read_bytes)
                        if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                            raise LedgerError("evidence_mismatch")
                        package = json.loads(data)
                        method = package["method"]
                        if (not isinstance(method["method_id"], str)
                                or not isinstance(method["resource_hashes"], list)):
                            raise LedgerError("evidence_mismatch")
                        package_id = row["name"].removeprefix("packages/").removesuffix(".json")
                        if package["package_id"] != package_id:
                            raise LedgerError("evidence_mismatch")
                        asset_ids = {package["atom_map"]["artifact_id"],
                                     package["qc"]["report"]["artifact_id"],
                                     *(item["artifact_id"] for item in package["assets"])}
                        package_sources[package_id] = (row["operation_id"], row["producer_kind"], asset_ids)
                        package_methods.append((method["method_id"], content_hash(method["resource_hashes"])))
                    except (KeyError, TypeError, ValueError, OSError) as error:
                        raise LedgerError("evidence_mismatch") from error
                if len(package_sources) != package_count:
                    raise LedgerError("evidence_mismatch")
                expected_assets = {f"assets/{pid}/{aid}" for pid, (_, _, ids) in package_sources.items()
                                   for aid in ids}
                asset_bindings = {row["name"]: row for row in bindings
                                  if row["name"].startswith("assets/")}
                if (len(asset_bindings) != sum(row["name"].startswith("assets/") for row in bindings)
                        or set(asset_bindings) != expected_assets):
                    raise LedgerError("evidence_mismatch")
                for name, row in asset_bindings.items():
                    _, pid, aid = name.split("/", 2)
                    operation_id, producer_kind, _ = package_sources[pid]
                    if (row.get("operation_id") != operation_id
                            or row.get("producer_kind") != producer_kind
                            or row.get("producer_path") != f"assets/{aid}"):
                        raise LedgerError("evidence_mismatch")
                actual = (sources[0], tuple(sorted(row["sha256"] for row in reviewed)),
                          tuple(sorted(package_methods))) if len(sources) == 1 else None
                profiles = _COMPONENT_SCOPES["stored-package-v1"]["profiles"]
                if actual not in {(row["source_sha256"], tuple(sorted(row["state_sha256s"])),
                                   tuple(sorted((method["method_id"], method["resource_sha256"])
                                                for method in row["package_methods"])))
                                   for row in profiles}:
                    raise LedgerError("capability_unavailable")
            count = await connection.fetchval(
                """SELECT count(*) FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND pool_id=$2 AND state IN ('queued','running')""",
                scope, self.pool_id)
            if count >= self.limits.max_queued_per_owner:
                raise LedgerError("admission_full")
            if self.limits.max_result_bytes > capacity - await self._used_capacity(connection):
                raise LedgerError("admission_full")
            if validated:
                now = await connection.fetchval("SELECT clock_timestamp()")
                operation_id = uuid4()
                operation = {
                    "schema_version": 1, "operation_id": str(operation_id),
                    "request_id": request["request_id"], "kind": request["kind"],
                    "revision": 1, "state": "queued", "request": request,
                    "request_sha256": digest, "receipt": None, "error": None,
                    "created_at": _stamp(now), "updated_at": _stamp(now),
                }
                validate(COMPONENT_SCHEMA, operation)
                await connection.execute(
                    """INSERT INTO neomd_service.preparation_operations
                    (operation_id,scope_id,request_id,pool_id,kind,request_sha256,request,snapshot,
                    source_upload_id,preparation_seed,state,revision,owner_id,deadline_at,reserved_output_bytes)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,NULL,$9,'queued',1,$10,$11,$12)""",
                    operation_id, scope, _uuid(request["request_id"]), self.pool_id,
                    request["kind"], digest, request, operation, secrets.randbelow(2147483647)+1,
                    self.owner_id, now + timedelta(seconds=self.limits.time_limit_seconds),
                    self.limits.max_result_bytes)
        if validated:
            self.signal.set()
            return operation
        from .component_preparation_execution import _validate_component_process
        validation_dir = self.root / "preparation-validation" / str(uuid4())
        try:
            input_sha = await self._materialize_component_inputs(bindings, validation_dir)
            job = {"kind": request["kind"], "input_dir": str(validation_dir),
                   "input_index_sha256": input_sha, "options": request["options"],
                   "request_id": request["request_id"],
                   "memory_limit_bytes": self.limits.memory_limit_bytes}
            outcome = await self._run_child(
                _validate_component_process, job,
                time.monotonic() + min(self.limits.time_limit_seconds, 600))
            if "error" in outcome:
                raise LedgerError(outcome["error"])
            if not outcome.get("ok"):
                raise LedgerError("verification_failed")
        finally:
            await asyncio.to_thread(shutil.rmtree, validation_dir, True)
        return await self._submit_component(scope_id, request, validated=True)

    async def _submit(self, scope_id, request, *, validated):
        topology = request.get("schema_version") == 2
        validate(TOPOLOGY_REQUEST if topology else PREPARATION_REQUEST, request)
        if self.closing or self.faulted or not self._initialized:
            raise LedgerError("service_unavailable")
        serialized = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        if len(serialized) > self.limits.max_request_bytes:
            raise LedgerError("payload_too_large")
        preset = self._topology_preset if topology else self._preset
        kind = TOPOLOGY_KIND if topology else KIND
        if (request["preset_id"] != preset["preset_id"]
                or request["preset_revision_sha256"] != preset["preset_revision_sha256"]):
            raise LedgerError("preset_revision_mismatch")
        source = request["source"]
        scope = _uuid(scope_id)
        digest = content_hash(request)
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            capacity = await connection.fetchval(
                "SELECT capacity_bytes FROM neomd_service.storage_capacity WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            if capacity is None:
                raise LedgerError("service_unavailable")
            existing = await connection.fetchrow(
                """SELECT request_sha256,snapshot FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND kind=$2 AND request_id=$3""",
                scope, kind, _uuid(request["request_id"]),
            )
            if existing is not None:
                if existing["request_sha256"] != digest:
                    raise LedgerError("request_conflict")
                return existing["snapshot"]
            upload = await connection.fetchrow(
                """SELECT receipt FROM neomd_service.preparation_source_uploads
                WHERE scope_id=$1 AND upload_id=$2 FOR UPDATE""",
                scope, _uuid(source["source_upload_id"]),
            )
            if upload is None:
                raise LedgerError("evidence_mismatch")
            receipt = upload["receipt"]
            source_sha = "content_sha256" if topology else "canonical_bytes_sha256"
            if (receipt["state"] != "stored"
                    or receipt["schema_version"] != (2 if topology else 1)
                    or receipt[source_sha] != source[source_sha]
                    or receipt["size_bytes"] != source["size_bytes"]):
                raise LedgerError("evidence_mismatch")
            if topology and (receipt["decoded_size_bytes"] != source["decoded_size_bytes"]
                             or receipt["topology_revision_id"] != source["topology_revision_id"]
                             or receipt["schema_id"] != source["schema_id"]):
                raise LedgerError("evidence_mismatch")
            referenced = await connection.fetchval(
                """SELECT EXISTS(SELECT 1 FROM neomd_service.preparation_operations
                WHERE source_upload_id=$1 AND state IN ('queued','running'))""",
                _uuid(source["source_upload_id"]),
            )
            if referenced:
                raise LedgerError("version_conflict")
            count = await connection.fetchval(
                """SELECT count(*) FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND pool_id=$2 AND state IN ('queued','running')""",
                scope, self.pool_id,
            )
            if count >= self.limits.max_queued_per_owner:
                raise LedgerError("admission_full")
            if self.limits.max_result_bytes > capacity - await self._used_capacity(connection):
                raise LedgerError("admission_full")
            binding = None
            if not topology:
                descriptor = source["descriptor"]
                binding = descriptor["sources"][0]
                if len(descriptor["sources"]) != 1 or binding["content_sha256"] != source[source_sha]:
                    raise LedgerError("evidence_mismatch")
            now = await connection.fetchval("SELECT clock_timestamp()")
            operation_id = uuid4()
            operation = {
                "schema_version": 2 if topology else 1,
                "operation_id": str(operation_id),
                "request_id": request["request_id"],
                "kind": kind,
                "revision": 1,
                "state": "queued",
                "request": request,
                "request_sha256": digest,
                "receipt": None,
                "error": None,
                "created_at": _stamp(now),
                "updated_at": _stamp(now),
            }
            validate(TOPOLOGY_SCHEMA if topology else SCHEMA, operation)
            seed = secrets.randbelow(2147483647) + 1
            deadline = now + timedelta(seconds=self.limits.time_limit_seconds)
            if validated:
                await connection.execute(
                    """INSERT INTO neomd_service.preparation_operations
                    (operation_id,scope_id,request_id,pool_id,kind,request_sha256,request,snapshot,
                    source_upload_id,preparation_seed,state,revision,owner_id,deadline_at,reserved_output_bytes)
                    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'queued',1,$11,$12,$13)""",
                    operation_id, scope, _uuid(request["request_id"]), self.pool_id, kind, digest,
                    request, operation, _uuid(source["source_upload_id"]), seed,
                    self.owner_id, deadline, self.limits.max_result_bytes,
                )
        if validated:
            self.signal.set()
            return operation
        # No visible/claimable intent exists until isolated validation completes.
        # Recheck upload, owner, admission and request identity before insertion.
        job = {
            "root": str(self.root), "upload_id": source["source_upload_id"],
            "format": source["format"],
            "memory_limit_bytes": self.limits.memory_limit_bytes,
        }
        if topology:
            job["reference"] = {key: source[key] for key in (
                "format", "schema_id", "content_sha256", "size_bytes",
                "decoded_size_bytes", "topology_revision_id")}
        else:
            job.update(canonical_bytes_sha256=source[source_sha], binding=binding)
        await self._validate_source(job, time.monotonic() + min(self.limits.time_limit_seconds, 600))
        return await self._submit(scope_id, request, validated=True)

    async def _validate_source(self, job, deadline):
        child = (_validate_topology_source_process if "reference" in job
                 else _validate_source_process)
        outcome = await self._run_child(child, job, deadline)
        if "error" in outcome:
            raise LedgerError(outcome["error"])
        if not outcome.get("ok"):
            raise LedgerError("unsupported_structure")

    async def get(self, scope_id, operation_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.preparation_operations WHERE scope_id=$1 AND operation_id=$2",
                _uuid(scope_id), _uuid(operation_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def get_topology(self, scope_id, operation_id):
        operation = await self.get(scope_id, operation_id)
        if operation["kind"] != TOPOLOGY_KIND:
            raise LedgerError("not_found")
        return operation

    async def get_protein(self, scope_id, operation_id):
        operation = await self.get(scope_id, operation_id)
        if operation["kind"] != KIND:
            raise LedgerError("not_found")
        return operation

    async def get_component(self, scope_id, operation_id):
        operation = await self.get(scope_id, operation_id)
        if operation["kind"] not in COMPONENT_KINDS:
            raise LedgerError("not_found")
        return operation

    async def component_by_request(self, scope_id, request_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                """SELECT snapshot FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND request_id=$2 AND kind LIKE 'component_%'""",
                _uuid(scope_id), _uuid(request_id))
        if value is None:
            raise LedgerError("not_found")
        return value

    async def cancel_component(self, scope_id, operation_id):
        await self.get_component(scope_id, operation_id)
        return await self.cancel(scope_id, operation_id)

    async def component_inventory(self, scope_id, operation_id):
        await self.get_component(scope_id, operation_id)
        return await self._result_document(scope_id, operation_id, "inventory")

    async def component_qc(self, scope_id, operation_id):
        await self.get_component(scope_id, operation_id)
        return await self._result_document(scope_id, operation_id, "qc")

    async def by_request(self, scope_id, request_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.preparation_operations WHERE scope_id=$1 AND kind=$2 AND request_id=$3",
                _uuid(scope_id), KIND, _uuid(request_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def topology_by_request(self, scope_id, request_id):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                "SELECT snapshot FROM neomd_service.preparation_operations WHERE scope_id=$1 AND kind=$2 AND request_id=$3",
                _uuid(scope_id), TOPOLOGY_KIND, _uuid(request_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def cancel_topology(self, scope_id, operation_id):
        await self.get_topology(scope_id, operation_id)
        return await self.cancel(scope_id, operation_id)

    async def cancel_protein(self, scope_id, operation_id):
        await self.get_protein(scope_id, operation_id)
        return await self.cancel(scope_id, operation_id)

    async def cancel(self, scope_id, operation_id):
        """Idempotent through the state machine; terminal operations conflict."""
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """SELECT * FROM neomd_service.preparation_operations
                WHERE scope_id=$1 AND operation_id=$2 FOR UPDATE""",
                _uuid(scope_id), _uuid(operation_id),
            )
            if row is None:
                raise LedgerError("not_found")
            if row["state"] == "queued":
                return await self._write(connection, row, "cancelled", code="preparation_terminal",
                                         message="Cancelled by owner request before execution.")
            if row["state"] == "running":
                await connection.execute(
                    """UPDATE neomd_service.preparation_operations
                    SET cancel_requested=true,updated_at=clock_timestamp() WHERE operation_id=$1""",
                    row["operation_id"],
                )
                self.signal.set()
                return await connection.fetchval(
                    "SELECT snapshot FROM neomd_service.preparation_operations WHERE operation_id=$1",
                    row["operation_id"],
                )
            raise LedgerError("preparation_terminal")

    # --- result access ---------------------------------------------------------

    async def inventory(self, scope_id, operation_id):
        await self.get_protein(scope_id, operation_id)
        return await self._result_document(scope_id, operation_id, "inventory")

    async def qc(self, scope_id, operation_id):
        await self.get_protein(scope_id, operation_id)
        return await self._result_document(scope_id, operation_id, "qc")

    async def _result_document(self, scope_id, operation_id, column):
        if self.faulted:
            raise LedgerError("service_unavailable")
        async with self.ledger.pool.acquire() as connection:
            value = await connection.fetchval(
                f"SELECT {column} FROM neomd_service.preparation_operations WHERE scope_id=$1 AND operation_id=$2",
                _uuid(scope_id), _uuid(operation_id),
            )
        if value is None:
            raise LedgerError("not_found")
        return value

    async def artifact_content(self, scope_id, operation_id, artifact_id):
        async with self.ledger.pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT pa.path,pa.size_bytes FROM neomd_service.preparation_artifacts pa
                JOIN neomd_service.preparation_operations po ON po.operation_id=pa.operation_id
                WHERE pa.scope_id=$1 AND pa.artifact_id=$2 AND po.operation_id=$3 AND po.scope_id=$1""",
                _uuid(scope_id), _uuid(artifact_id), _uuid(operation_id),
            )
        if row is None:
            raise LedgerError("not_found")
        if row["size_bytes"] > self.limits.max_result_bytes:
            raise LedgerError("payload_too_large")
        path = self.root / "preparations" / str(_uuid(operation_id)) / row["path"]
        data = await asyncio.to_thread(path.read_bytes)
        if len(data) != row["size_bytes"]:
            raise LedgerError("integrity_mismatch")
        return data

    # --- queue runner ------------------------------------------------------------

    async def _write(self, connection, row, state, *, code=None, message=None, receipt=None):
        now = await connection.fetchval("SELECT clock_timestamp()")
        snapshot = dict(row["snapshot"])
        snapshot.update(state=state, revision=row["revision"] + 1, updated_at=_stamp(now))
        if receipt is not None:
            snapshot["receipt"] = receipt
        if code is not None:
            snapshot["error"] = {
                "code": code, "message": message or code, "retryable": False,
                "request_id": snapshot["request_id"],
            }
        validate(COMPONENT_SCHEMA if snapshot["kind"] in COMPONENT_KINDS else
                 TOPOLOGY_SCHEMA if snapshot["schema_version"] == 2 else SCHEMA, snapshot)
        if row["state"] == "queued" and state not in PENDING:
            # A never-claimed operation owns no output writer or staging files.
            await connection.execute(
                "UPDATE neomd_service.preparation_operations SET reserved_output_bytes=0 WHERE operation_id=$1",
                row["operation_id"],
            )
        await connection.execute(
            """UPDATE neomd_service.preparation_operations
            SET state=$2,revision=$3,snapshot=$4,updated_at=$5 WHERE operation_id=$1""",
            row["operation_id"], state, snapshot["revision"], snapshot, now,
        )
        return snapshot

    async def _claim(self):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            await connection.fetchval(
                "SELECT max_queued_per_owner FROM neomd_service.preparation_policy WHERE pool_id=$1 FOR UPDATE",
                self.pool_id,
            )
            stale = await connection.fetch(
                """SELECT * FROM neomd_service.preparation_operations WHERE pool_id=$1
                AND state IN ('queued','running') AND deadline_at<=clock_timestamp()
                ORDER BY deadline_at LIMIT 8 FOR UPDATE SKIP LOCKED""",
                self.pool_id,
            )
            for row in stale:
                await self._write(connection, row, "failed", code="preparation_time_limit_exceeded")
            row = await connection.fetchrow(
                """SELECT *,clock_timestamp() AS claimed_at FROM neomd_service.preparation_operations
                WHERE pool_id=$1 AND owner_id=$2 AND state='queued' AND deadline_at>clock_timestamp()
                ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED""",
                self.pool_id, self.owner_id,
            )
            if row is None:
                return None
            snapshot = await self._write(connection, row, "running")
            lease_id = uuid4()
            await connection.execute(
                """UPDATE neomd_service.preparation_operations SET lease_id=$2,lease_expires_at=deadline_at
                WHERE operation_id=$1""",
                row["operation_id"], lease_id,
            )
            return dict(row, lease_id=lease_id, snapshot=snapshot, revision=snapshot["revision"])

    async def _run_child(self, target, job, deadline, *, cancel_check=None):
        if self.closing or self.faulted:
            raise LedgerError("service_unavailable")
        context = multiprocessing.get_context("spawn")
        result, child_result = Channel.pair(max_message_bytes=1048576)
        process = context.Process(target=target, args=(job, child_result, os.getpid()))
        try:
            process.start()
        except Exception as error:
            result.close()
            child_result.close()
            process.close()
            raise LedgerError("service_unavailable") from error
        child_result.close()
        self._children.add(process)
        try:
            received = None
            while True:
                try:
                    message = result.get_nowait()
                    if received is not None or not isinstance(message, dict):
                        raise LedgerError("integrity_mismatch")
                    received = message
                except queue.Empty:
                    pass
                if received is not None:
                    break
                if cancel_check is not None and await cancel_check():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                    raise LedgerError("preparation_terminal")
                if not process.is_alive():
                    await asyncio.sleep(0.05)
                    try:
                        message = result.get_nowait()
                        if isinstance(message, dict):
                            received = message
                    except queue.Empty:
                        pass
                    break
                if time.monotonic() >= deadline:
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                    raise LedgerError("preparation_time_limit_exceeded")
                await asyncio.sleep(0.05)
            # A child may publish its bounded result before the interpreter
            # finishes releasing OpenMM/RDKit resources. Wait for clean exit
            # within the existing operation deadline instead of declaring a
            # successful scientific result lost after an arbitrary one second.
            while process.is_alive() and time.monotonic() < deadline:
                if cancel_check is not None and await cancel_check():
                    process.kill()
                    await asyncio.to_thread(process.join, 5)
                    raise LedgerError("preparation_terminal")
                await asyncio.to_thread(process.join, min(0.25, max(0, deadline - time.monotonic())))
            if process.is_alive():
                raise LedgerError("preparation_time_limit_exceeded")
            if received is None or process.exitcode != 0:
                raise LedgerError("worker_lost")
            return received
        finally:
            result.close()
            self._children.discard(process)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 0.5)
            process.close()

    async def _cancel_requested(self, operation_id):
        async with self.ledger.pool.acquire() as connection:
            return await connection.fetchval(
                "SELECT cancel_requested FROM neomd_service.preparation_operations WHERE operation_id=$1",
                operation_id,
            )

    async def _finish(self, claimed, state, *, code=None, message=None, receipt=None,
                      inventory=None, qc=None, artifacts=(), release_reservation=False):
        async with self.ledger.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM neomd_service.preparation_operations WHERE operation_id=$1 FOR UPDATE",
                claimed["operation_id"],
            )
            if row["owner_id"] != self.owner_id or row["lease_id"] != claimed["lease_id"]:
                raise LedgerError("lease_expired")
            if row["state"] not in PENDING:
                return row["snapshot"]
            if state == "succeeded":
                if row["cancel_requested"]:
                    raise LedgerError("preparation_terminal")
                if row["deadline_at"] <= await connection.fetchval("SELECT clock_timestamp()"):
                    raise LedgerError("preparation_time_limit_exceeded")
            if sum(artifact["size_bytes"] for artifact in artifacts) > row["reserved_output_bytes"]:
                raise LedgerError("result_too_large")
            if release_reservation:
                await connection.execute(
                    "UPDATE neomd_service.preparation_operations SET reserved_output_bytes=0 WHERE operation_id=$1",
                    row["operation_id"],
                )
            for artifact in artifacts:
                await connection.execute(
                    """INSERT INTO neomd_service.preparation_artifacts
                    (artifact_id,operation_id,scope_id,role,path,sha256,size_bytes)
                    VALUES($1,$2,$3,$4,$5,$6,$7)""",
                    _uuid(artifact["artifact_id"]), row["operation_id"], row["scope_id"],
                    artifact["role"], artifact["path"], artifact["sha256"], artifact["size_bytes"],
                )
            if receipt is not None or inventory is not None or qc is not None:
                await connection.execute(
                    """UPDATE neomd_service.preparation_operations
                    SET receipt=$2,inventory=$3,qc=$4 WHERE operation_id=$1""",
                    row["operation_id"], receipt, inventory, qc,
                )
            return await self._write(connection, row, state, code=code, message=message,
                                     receipt=receipt)

    def _staging(self, operation_id):
        return self.root / "preparation-staging" / str(operation_id)

    def _final(self, operation_id):
        return self.root / "preparations" / str(operation_id)

    async def _execute(self, claimed):
        if claimed["kind"] in COMPONENT_KINDS:
            await self._execute_component(claimed)
            return
        operation_id = claimed["operation_id"]
        request = claimed["request"]
        deadline = time.monotonic() + max(
            0, (claimed["deadline_at"] - claimed["claimed_at"]).total_seconds())
        staging = self._staging(operation_id)
        await asyncio.to_thread(shutil.rmtree, staging, True)
        source = request["source"]
        topology = request["schema_version"] == 2
        job = {
            "root": str(self.root), "upload_id": source["source_upload_id"],
            "format": source["format"],
            "staging": str(staging),
            "parameters": request["parameters"],
            "system_parameters": SYSTEM_PARAMETERS,
            "seed": claimed["preparation_seed"],
            "memory_limit_bytes": self.limits.memory_limit_bytes,
            "max_result_bytes": self.limits.max_result_bytes,
        }
        if topology:
            job["reference"] = {key: source[key] for key in (
                "format", "schema_id", "content_sha256", "size_bytes",
                "decoded_size_bytes", "topology_revision_id")}
        else:
            job.update(canonical_bytes_sha256=source["canonical_bytes_sha256"],
                       binding=source["descriptor"]["sources"][0])

        async def cancelled():
            return await self._cancel_requested(operation_id)

        try:
            prepare_child = _prepare_topology_process if topology else _prepare_process
            prepared = await self._run_child(prepare_child, job, deadline,
                                             cancel_check=cancelled)
            if "error" in prepared:
                code = "verification_failed" if prepared["error"] == "unsupported_structure" \
                    else prepared["error"]
                message = ("Covalently attached ligand needs a qualified joint template."
                           if topology and code == "unsupported_relation" else None)
                await self._fail(claimed, code, staging, message=message)
                return
            verify_job = {
                "staging": str(staging),
                "prepared_revision_id": prepared["prepared_revision_id"],
                "manifest_sha256": prepared["manifest_sha256"],
                "memory_limit_bytes": self.limits.memory_limit_bytes,
            }
            verify_child = _verify_topology_process if topology else _verify_process
            verified = await self._run_child(verify_child, verify_job, deadline,
                                             cancel_check=cancelled)
            if "error" in verified:
                await self._fail(claimed, verified["error"], staging)
                return
            if verified["prepared_revision_id"] != prepared["prepared_revision_id"]:
                await self._fail(claimed, "verification_failed", staging)
                return
            if topology:
                await self._publish_topology(claimed, prepared, verified, staging)
            else:
                await self._publish(claimed, prepared, verified, staging)
        except LedgerError as error:
            if error.code == "preparation_terminal":
                await self._fail(claimed, "preparation_terminal", staging, state="cancelled",
                                 message="Cancelled by owner request; staged artifacts were discarded.")
            elif error.code == "worker_lost":
                await self._fail(claimed, "worker_lost", staging,
                                 message="The isolated executor exited without a verifiable result.")
            elif error.code in {"preparation_time_limit_exceeded", "preparation_memory_limit_exceeded",
                                "result_too_large", "protonation_conflict", "verification_failed",
                                "evidence_mismatch", "integrity_mismatch",
                                "connectivity_incomplete", "invalid_topology_record"}:
                await self._fail(claimed, "verification_failed" if error.code == "integrity_mismatch"
                                 else error.code, staging)
            else:
                raise
        finally:
            await asyncio.to_thread(shutil.rmtree, staging, True)

    async def _execute_component(self, claimed):
        from .component_preparation_execution import (
            _prepare_component_process,
            _verify_component_process,
        )

        operation_id = claimed["operation_id"]
        staging = self._staging(operation_id)
        await asyncio.to_thread(shutil.rmtree, staging, True)
        deadline = time.monotonic() + max(
            0, (claimed["deadline_at"] - claimed["claimed_at"]).total_seconds())

        async def cancelled():
            return await self._cancel_requested(operation_id)

        try:
            async with self.ledger.pool.acquire() as connection:
                bindings = await self._component_bindings(
                    connection, claimed["scope_id"], claimed["request"]["inputs"])
            input_dir = staging / "inputs"
            input_sha = await self._materialize_component_inputs(bindings, input_dir)
            job = {
                "kind": claimed["kind"], "input_dir": str(input_dir),
                "input_index_sha256": input_sha, "staging": str(staging),
                "options": claimed["request"]["options"],
                "request_id": claimed["request"]["request_id"],
                "memory_limit_bytes": self.limits.memory_limit_bytes,
                "max_result_bytes": self.limits.max_result_bytes,
            }
            prepared = await self._run_child(
                _prepare_component_process, job, deadline, cancel_check=cancelled)
            if "error" in prepared:
                await self._fail(claimed, prepared["error"], staging)
                return
            if not prepared.get("ok") or prepared.get("kind") != claimed["kind"]:
                await self._fail(claimed, "verification_failed", staging)
                return
            verified = await self._run_child(
                _verify_component_process, dict(job, prepared=prepared), deadline,
                cancel_check=cancelled)
            if "error" in verified:
                await self._fail(claimed, verified["error"], staging)
                return
            if (not verified.get("ok") or verified.get("kind") != claimed["kind"]
                    or verified.get("primary_sha256") != prepared["primary_sha256"]
                    or verified.get("primary_id") != prepared["primary_id"]):
                await self._fail(claimed, "verification_failed", staging)
                return
            await self._publish_component(claimed, prepared, verified, bindings, staging)
        except LedgerError as error:
            if error.code == "preparation_terminal":
                await self._fail(claimed, "preparation_terminal", staging, state="cancelled",
                                 message="Cancelled by owner request; staged artifacts were discarded.")
            elif error.code in {"worker_lost", "preparation_time_limit_exceeded",
                                "preparation_memory_limit_exceeded", "result_too_large",
                                "evidence_mismatch", "integrity_mismatch",
                                "verification_failed", "capability_unavailable"}:
                await self._fail(claimed, error.code, staging)
            else:
                raise
        finally:
            await asyncio.to_thread(shutil.rmtree, staging, True)

    async def _salvage_log(self, claimed, staging):
        """Failed/cancelled operations retain only their public-safe log artifact."""
        source = staging / "preparation.log"
        if not source.is_file():
            return []
        final = self._final(claimed["operation_id"])
        if final.exists():
            raise LedgerError("integrity_mismatch")
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(final.mkdir)
        target = final / "preparation.log"
        await asyncio.to_thread(os.rename, source, target)
        await asyncio.to_thread(sync_directories, final, root=self.root)
        return [{
            "artifact_id": str(uuid4()), "role": "log", "path": "preparation.log",
            "sha256": await asyncio.to_thread(_digest, target),
            "size_bytes": (await asyncio.to_thread(target.lstat)).st_size,
        }]

    async def _fail(self, claimed, code, staging, *, state="failed", message=None):
        try:
            artifacts = await self._salvage_log(claimed, staging)
        except LedgerError:
            artifacts = []
        # Release only after the child has exited and owned staging is removed.
        if staging.exists():
            await asyncio.to_thread(shutil.rmtree, staging)
        release = bool(artifacts) or not self._final(claimed["operation_id"]).exists()
        await self._finish(claimed, state, code=code, message=message, artifacts=artifacts,
                           release_reservation=release)

    async def _publish_component(self, claimed, prepared, verified, bindings, staging):
        result_dir = staging / "result"
        final = self._final(claimed["operation_id"])
        if final.exists() or not result_dir.is_dir():
            raise LedgerError("integrity_mismatch")
        declared = {}
        for row in (*prepared.get("artifacts", ()), *verified.get("artifacts", ())):
            if row["path"] in declared and declared[row["path"]] != row:
                raise LedgerError("integrity_mismatch")
            declared[row["path"]] = row
        if not declared:
            raise LedgerError("verification_failed")
        if any(path.is_symlink() for path in result_dir.rglob("*")):
            raise LedgerError("integrity_mismatch")
        paths = {path.relative_to(result_dir).as_posix(): path
                 for path in result_dir.rglob("*") if path.is_file()}
        if set(paths) != set(declared):
            raise LedgerError("integrity_mismatch")
        artifacts = []
        for name, path in sorted(paths.items()):
            row = declared[name]
            if (_digest(path) != row["sha256"]
                    or path.lstat().st_size != row["size_bytes"]):
                raise LedgerError("integrity_mismatch")
            role = ("state" if name == "state.json" else
                    "instance" if name == "instance.json" else
                    "package" if name == "package.json" else
                    "qc" if name == "qc-report.json" else
                    "prepared" if name.startswith("prepared/") else
                    "assembly" if name.startswith("assembly-") or name in
                    {"topology.json", "positions.f64le", "system.xml"} else
                    "report" if name.endswith("report.json") else "asset")
            artifacts.append({"artifact_id": str(uuid4()), "role": role, "path": name,
                              "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
        log_source = staging / "preparation.log"
        if log_source.is_file():
            artifacts.append({"artifact_id": str(uuid4()), "role": "log",
                              "path": "preparation.log", "sha256": _digest(log_source),
                              "size_bytes": log_source.lstat().st_size})
        if sum(row["size_bytes"] for row in artifacts) > claimed["reserved_output_bytes"]:
            raise LedgerError("result_too_large")
        now = _stamp(datetime.now(timezone.utc))
        kind = claimed["kind"]
        input_bindings = []
        for row in bindings:
            source = ({"upload_id": row["upload_id"]} if "upload_id" in row else
                      {"operation_id": row["operation_id"],
                       "artifact_id": row["artifact_id"],
                       "producer_receipt_sha256": row["producer_receipt_sha256"]})
            input_bindings.append({"name": row["name"], "sha256": row["sha256"],
                                   "size_bytes": row["size_bytes"],
                                   "resolved_owner_scope": row["resolved_owner_scope"],
                                   "source": source})
        report_sha = verified.get("report_sha256")
        if report_sha not in {row["sha256"] for row in artifacts}:
            raise LedgerError("verification_failed")
        receipt = {
            "schema_version": 1, "receipt_id": str(uuid4()),
            "operation_id": str(claimed["operation_id"]),
            "request_id": str(claimed["request_id"]), "kind": kind,
            "owner_scope": str(claimed["scope_id"]),
            "backend": {"provider": "neodynamics", "pool_id": self.pool_id},
            "input_bindings": input_bindings,
            "result": {"object_type": ("review_record" if kind == "component_review" else
                                       "parameter_package" if kind in
                                       {"component_generate", "component_import"} else
                                       "prepared_bundle"),
                       "id": verified.get("result_id", prepared["primary_id"]),
                       "sha256": verified.get("result_sha256", prepared["primary_sha256"])},
            "verification": {"profile": f"{kind}_v1",
                             "verifier": "neomd_server.component_preparation_execution",
                             "report_sha256": report_sha, "verified_at": now},
            "artifacts": [{key: row[key] for key in
                           ("artifact_id", "role", "sha256", "size_bytes")}
                          for row in artifacts],
            "created_at": now,
        }
        validate("urn:neodyn:component-operations:v1:receipt", receipt)
        inventory = {"schema_version": 1, "operation_id": str(claimed["operation_id"]),
                     "receipt_id": receipt["receipt_id"],
                     "artifacts": artifacts,
                     "total_size_bytes": sum(row["size_bytes"] for row in artifacts)}
        validate("urn:neodyn:component-operations:v1:inventory", inventory)
        qc = {"schema_version": 1, "operation_id": str(claimed["operation_id"]),
              "receipt_id": receipt["receipt_id"],
              "outcome": "passed", "report_sha256": report_sha,
              "summary": json.dumps(verified.get("qc", {}), sort_keys=True)[:1024],
              "blockers": []}
        validate("urn:neodyn:component-operations:v1:qc", qc)
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(os.rename, result_dir, final)
        if log_source.is_file():
            await asyncio.to_thread(os.rename, log_source, final / "preparation.log")
        await asyncio.to_thread(sync_directories, final, root=self.root)
        try:
            await self._finish(claimed, "succeeded", receipt=receipt, inventory=inventory,
                               qc=qc, artifacts=artifacts, release_reservation=True)
        except LedgerError as error:
            if error.code in {"preparation_terminal", "preparation_time_limit_exceeded"}:
                await asyncio.to_thread(shutil.rmtree, final)
            raise

    async def _publish(self, claimed, prepared, verified, staging):
        operation_id = claimed["operation_id"]
        bundle = staging / prepared["prepared_revision_id"]
        final = self._final(operation_id)
        if final.exists():
            raise LedgerError("integrity_mismatch")
        artifacts = []
        for item in prepared["artifacts"]:
            path = bundle / item["path"]
            digest = await asyncio.to_thread(_digest, path)
            size = (await asyncio.to_thread(path.lstat)).st_size
            if digest != item["sha256"] or size != item["size_bytes"]:
                raise LedgerError("integrity_mismatch")
            artifacts.append({**item, "role": _role(item["path"]), "artifact_id": str(uuid4())})
        log_source = staging / "preparation.log"
        if log_source.is_file():
            artifacts.append({
                "path": "preparation.log", "sha256": await asyncio.to_thread(_digest, log_source),
                "size_bytes": (await asyncio.to_thread(log_source.lstat)).st_size,
                "role": "log", "artifact_id": str(uuid4()),
            })
        if sum(item["size_bytes"] for item in artifacts) > claimed["reserved_output_bytes"]:
            raise LedgerError("result_too_large")
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(os.rename, bundle, final)
        if log_source.is_file():
            await asyncio.to_thread(os.rename, log_source, final / "preparation.log")
        await asyncio.to_thread(sync_directories, final, root=self.root)
        receipt = self._receipt(claimed, prepared, verified)
        inventory = {
            "schema_version": 1,
            "operation_id": str(operation_id),
            "receipt_id": receipt["receipt_id"],
            "prepared_revision_id": prepared["prepared_revision_id"],
            "artifacts": [
                {key: artifact[key] for key in ("artifact_id", "role", "path", "sha256", "size_bytes")}
                for artifact in artifacts
            ],
            "total_size_bytes": sum(artifact["size_bytes"] for artifact in artifacts),
        }
        validate(INVENTORY, inventory)
        qc = self._qc(claimed, receipt, verified)
        await asyncio.to_thread(shutil.rmtree, staging)
        try:
            await self._finish(claimed, "succeeded", receipt=receipt, inventory=inventory, qc=qc,
                               artifacts=artifacts, release_reservation=True)
        except LedgerError as error:
            if error.code in {"preparation_terminal", "preparation_time_limit_exceeded"}:
                # No database publication occurred; only this operation owns final.
                await asyncio.to_thread(shutil.rmtree, final)
            raise

    async def _publish_topology(self, claimed, prepared, verified, staging):
        """Durably publish a verified explicit-topology result for its owner."""
        operation_id = claimed["operation_id"]
        bundle = staging / "prepared" / prepared["prepared_revision_id"]
        final = self._final(operation_id)
        if final.exists():
            raise LedgerError("integrity_mismatch")
        artifacts = []
        for item in prepared["artifacts"]:
            path = bundle / item["path"]
            digest = await asyncio.to_thread(_digest, path)
            size = (await asyncio.to_thread(path.lstat)).st_size
            if digest != item["sha256"] or size != item["size_bytes"]:
                raise LedgerError("integrity_mismatch")
            artifacts.append({**item, "role": _role(item["path"]),
                              "artifact_id": str(uuid4())})
        if sum(item["size_bytes"] for item in artifacts) > claimed["reserved_output_bytes"]:
            raise LedgerError("result_too_large")
        request = claimed["request"]
        now = _stamp(datetime.now(timezone.utc))
        parameters = dict(self._topology_preset["accepted_parameters"])
        parameters.update(request["parameters"])
        receipt = {
            "schema_version": 2,
            "receipt_id": str(uuid4()),
            "operation_id": str(operation_id),
            "request_id": request["request_id"],
            "owner_scope": str(claimed["scope_id"]),
            "backend": {"provider": "neodynamics", "pool_id": self.pool_id},
            "source": request["source"],
            "preset": {"preset_id": self._topology_preset["preset_id"],
                       "preset_revision_sha256": self._topology_preset["preset_revision_sha256"]},
            "parameters": parameters,
            "provenance": {
                "versions": verified["versions"],
                "openmm_build": prepared["openmm_build"],
                "platform": "CPU",
                "resources": self._topology_preset["resources"],
                "wall_time_seconds": prepared["wall_time_seconds"],
                "limits_applied": {
                    "max_source_bytes": self.limits.max_source_bytes,
                    "max_result_bytes": self.limits.max_result_bytes,
                    "time_limit_seconds": self.limits.time_limit_seconds,
                    "memory_limit_bytes": self.limits.memory_limit_bytes,
                },
            },
            "verification": {
                "profile": "explicit_topology_full_preparation_v2",
                "verifier": "neomd.topology_preparation.verify_explicit_topology_preparation",
                "prepared_revision_id": verified["prepared_revision_id"],
                "manifest_sha256": prepared["manifest_sha256"],
                "mapping_sha256": verified["mapping_sha256"],
                "changes_sha256": verified["changes_sha256"],
                "verified_at": now,
            },
            "artifacts": [{key: row[key] for key in
                           ("artifact_id", "role", "sha256", "size_bytes")}
                          for row in artifacts],
            "qc": verified["qc"],
            "created_at": now,
        }
        validate("urn:neodyn:protein-preparation:v2:receipt", receipt)
        await asyncio.to_thread(self.root.joinpath("preparations").mkdir, exist_ok=True)
        await asyncio.to_thread(os.rename, bundle, final)
        await asyncio.to_thread(sync_directories, final, root=self.root)
        await asyncio.to_thread(shutil.rmtree, staging)
        try:
            await self._finish(claimed, "succeeded", receipt=receipt, artifacts=artifacts,
                               release_reservation=True)
        except LedgerError as error:
            if error.code in {"preparation_terminal", "preparation_time_limit_exceeded"}:
                await asyncio.to_thread(shutil.rmtree, final)
            raise

    def _receipt(self, claimed, prepared, verified):
        request = claimed["request"]
        source = request["source"]
        parameters = dict(self._preset["accepted_parameters"])
        parameters.update(request["parameters"])
        now = _stamp(datetime.now(timezone.utc))
        receipt = {
            "schema_version": 1,
            "receipt_id": str(uuid4()),
            "operation_id": str(claimed["operation_id"]),
            "request_id": request["request_id"],
            "owner_scope": str(claimed["scope_id"]),
            "backend": {"provider": "neodynamics", "pool_id": self.pool_id},
            "source": {
                "format": source["format"],
                "canonical_bytes_sha256": source["canonical_bytes_sha256"],
                "size_bytes": source["size_bytes"],
                "identity": source["identity"],
            },
            "preset": {
                "preset_id": self._preset["preset_id"],
                "preset_revision_sha256": self._preset["preset_revision_sha256"],
            },
            "parameters": parameters,
            "provenance": {
                "versions": verified["versions"],
                "openmm_build": prepared["openmm_build"],
                "platform": "CPU",
                "preparation_seed": claimed["preparation_seed"],
                "resources": self._preset["resources"],
                "wall_time_seconds": prepared["wall_time_seconds"],
                "limits_applied": {
                    "max_source_bytes": self.limits.max_source_bytes,
                    "max_result_bytes": self.limits.max_result_bytes,
                    "time_limit_seconds": self.limits.time_limit_seconds,
                    "memory_limit_bytes": self.limits.memory_limit_bytes,
                },
            },
            "verification": {
                "profile": "canonical_protein_full_preparation_v1",
                "verifier": "neomd.prepared_pipeline.verify_protein_preparation",
                "prepared_revision_id": verified["prepared_revision_id"],
                "manifest_sha256": prepared["manifest_sha256"],
                "mapping_sha256": verified["mapping_sha256"],
                "assembly_report_sha256": verified["assembly_report_sha256"],
                "verified_at": now,
            },
            "created_at": now,
        }
        validate(RECEIPT, receipt)
        return receipt

    def _qc(self, claimed, receipt, verified):
        request = claimed["request"]
        defaults = self._preset["accepted_parameters"]
        sources = {
            name: "preset_default" if request["parameters"][name] == defaults[name] else "request_override"
            for name in ("ph", "ionic_strength_mol_per_l", "padding_nm")
        }
        summary = dict(verified["qc"])
        box = summary.pop("box_vectors_nm")
        qc = {
            "schema_version": 1,
            "operation_id": str(claimed["operation_id"]),
            "receipt_id": receipt["receipt_id"],
            "outcome": "passed",
            "summary": summary,
            "box_vectors_nm": box,
            "parameter_sources": sources,
            "blockers": [],
        }
        validate(QC, qc)
        return qc

    async def _run(self):
        while not self.closing and not self.faulted:
            claimed = None
            try:
                claimed = await self._claim()
                if claimed is None:
                    self.signal.clear()
                    try:
                        await asyncio.wait_for(self.signal.wait(), 1)
                    except TimeoutError:
                        pass
                    continue
                await self._execute(claimed)
            except LedgerError as error:
                if claimed is not None:
                    try:
                        await self._finish(claimed, "failed", code="service_unavailable")
                    except LedgerError as finish_error:
                        if finish_error.code != "lease_expired":
                            self.faulted = True
                elif error.code in {"service_unavailable", "worker_lost"}:
                    await asyncio.sleep(0.2)
                else:
                    self.faulted = True
            except asyncio.CancelledError:
                raise
            except BaseException:
                self.faulted = True
