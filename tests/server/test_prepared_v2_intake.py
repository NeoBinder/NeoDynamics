"""Real spawned intake of versioned identity fixtures, not assembly qualification."""

import asyncio
import hashlib
import json
import struct
from pathlib import Path
from uuid import uuid4

import pytest
from input_helpers import commit_bundle
from test_assets import store, upload

from neomd_server.contract import validate
from neomd_server.ledger import LedgerError
from neomd_server.topology import COLLECTIONS, parse_topology_bytes

CONTRACTS = Path(__file__).parents[2] / "contracts/prepared-bundle"


def repair_fixture_bytes():
    """Original repair fixture stays unsupported, never relabelled as load-only."""
    directory = CONTRACTS / "v2/fixtures/standard-heavy-repair"
    manifest = json.loads((directory / "manifest.json").read_text())
    parts = {name: (directory / name).read_bytes()
             for name in ("manifest.json", *manifest["artifacts"])}
    manifest["parameters"].update(system_constraints=0, force_types=[])
    parts["manifest.json"] = json.dumps(manifest, allow_nan=False).encode()
    return parts


def fixture_bytes(version, tmp_path):
    if version == 2:
        return load_fixture_bytes(tmp_path)
    assert version == 1
    directory = CONTRACTS / "v1/fixtures/water-extra-site"
    manifest = json.loads((directory / "manifest.json").read_text())
    return {name: (directory / name).read_bytes()
            for name in ("manifest.json", *manifest["artifacts"])}


def load_fixture_bytes(tmp_path):
    """Publish the original four source atoms without repair or assembly."""
    import openmm
    from openmm import unit

    from neomd.preparation import load_protein_source
    from neomd.prepared import PreparationLineage

    fixture = CONTRACTS / "v2/fixtures/standard-heavy-repair"
    old = json.loads((fixture / "manifest.json").read_text())
    source_path = fixture / old["sources"][0]["artifact"]
    descriptor = json.loads((CONTRACTS / "v2/fixtures/source-input.json").read_text())["sources"][0]
    # This is a new actual load by the current test producer. No frozen file or
    # historical bundle metadata is changed; its original source graph is reused.
    descriptor["loader"]["version"] = openmm.__version__
    source = load_protein_source(source_path, descriptor)
    lineage = PreparationLineage({"schema_version": 2, "sources": [descriptor]})
    lineage.append_source(source.topology, source.positions_nm, input_key="protein",
                          kind="structure", loader=source.loader, source_path=source_path)
    system, particles = openmm.System(), []
    for atom in source.topology.atoms():
        mass = atom.element.mass.value_in_unit(unit.dalton)
        system.addParticle(mass)
        particles.append({"mass_dalton": mass, "virtual_site": None})
    xml = tmp_path / "load-inventory.xml"
    xml.write_text(openmm.XmlSerializer.serialize(system))
    manifest_path = Path(lineage.publish(
        tmp_path / "load-fixtures", topology=source.topology, positions_nm=source.positions_nm,
        particles=particles, coordinates_nm=source.positions_nm, box_vectors_nm=None,
        artifacts={"system.xml": xml, "topology.pdbx": source_path},
        parameters={"system_constraints": 0, "force_types": [], "not_for_simulation": True}))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["topology_atom_count"] == manifest["particle_count"] == 4
    assert [step["operation"] for step in manifest["preparation_steps"]] == ["load"]
    return {name: (manifest_path.parent / name).read_bytes()
            for name in ("manifest.json", *manifest["artifacts"])}


@pytest.mark.parametrize("version", [1, 2])
def test_versioned_bundle_commits_exact_source_mapping_and_projection(database_dsn, tmp_path, version):
    async def scenario():
        parts = fixture_bytes(version, tmp_path)
        manifest = json.loads(parts["manifest.json"])
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as inputs:
            uploaded = await upload(inputs, scope, parts)
            committed = await commit_bundle(inputs, scope, uploaded["upload_id"])
            assert committed["state"] == "validated"
            prepared = committed["prepared"]
            assert prepared["manifest_sha256"] == hashlib.sha256(parts["manifest.json"]).hexdigest()
            assert prepared["mapping_sha256"] == hashlib.sha256(parts["mapping.json"]).hexdigest()
            assert prepared["prepared_revision_id"] == manifest["prepared_revision_id"]
            asset = await inputs.prepared(scope, prepared["bundle_id"])
            assert asset["particle_count"] == manifest["particle_count"]
            assert asset["topology_atom_count"] == manifest["topology_atom_count"]
            for name, data in parts.items():
                assert (tmp_path / "prepared" / prepared["bundle_id"] / name).read_bytes() == data
            descriptor = parse_topology_bytes(
                await inputs.topology(scope, prepared["bundle_id"]), limit=1048576)
            validate("urn:neodyn:topology-view:v1:view", descriptor)
            assert descriptor["prepared_schema_version"] == version
            rows = {}
            for collection in descriptor["collections"]:
                rows[collection["kind"]] = []
                for page in collection["pages"]:
                    data = await inputs.topology_page(scope, prepared["bundle_id"],
                        descriptor["view_sha256"], collection["kind"], page["page_index"])
                    assert len(data) == page["size_bytes"]
                    assert hashlib.sha256(data).hexdigest() == page["content_sha256"]
                    value = parse_topology_bytes(data, limit=1048576)
                    validate("urn:neodyn:topology-view:v1:page", value)
                    assert value["prepared_schema_version"] == version
                    rows[collection["kind"]].extend(value["rows"])
            assert rows == {kind: manifest[kind] for kind in COLLECTIONS}
            assert await commit_bundle(inputs, scope, uploaded["upload_id"]) == committed
        # Registration and the complete projection survive an InputStore owner change.
        async with store(database_dsn, tmp_path) as reopened:
            assert await reopened.prepared(scope, prepared["bundle_id"]) == asset
            assert parse_topology_bytes(await reopened.topology(scope, prepared["bundle_id"]),
                                        limit=1048576) == descriptor

    asyncio.run(scenario())


def rejected_parts(fault, tmp_path):
    parts = load_fixture_bytes(tmp_path)
    manifest, mapping = (json.loads(parts[name]) for name in ("manifest.json", "mapping.json"))
    if fault == "unknown_manifest_version":
        manifest["schema_version"] = 3
    elif fault == "boolean_manifest_version":
        manifest["schema_version"] = True
    elif fault == "floating_manifest_version":
        manifest["schema_version"] = 2.0
    elif fault == "unknown_mapping_version":
        mapping["schema_version"] = 3
    elif fault == "boolean_mapping_version":
        mapping["schema_version"] = True
    elif fault == "floating_mapping_version":
        mapping["schema_version"] = 2.0
    elif fault == "mixed_versions":
        mapping["schema_version"] = 1
    elif fault.startswith("missing_source_"):
        del manifest["sources"][0][fault.removeprefix("missing_source_")]
    elif fault == "source_bonds_removed":
        manifest["sources"][0]["bonds"] = []
    elif fault == "source_residue_name_changed":
        manifest["sources"][0]["residue_bindings"][0]["name"] = "GLY"
    elif fault == "source_polymer_chain_changed":
        manifest["sources"][0]["polymer_sequences"][0]["input_chain_index"] = 9
    elif fault == "relabelled_source":
        manifest["sources"][0]["kind"] = "ligand"
    elif fault == "extra_load_step":
        manifest["preparation_steps"].append(manifest["preparation_steps"][0].copy())
    elif fault == "translation_history":
        manifest["preparation_steps"].append({"operation": "translate", "translation_nm": [0, 0, 0]})
    elif fault == "moved_f64_coordinate":
        name = "coordinates.f64le"
        first = struct.unpack("<d", parts[name][:8])[0]
        parts[name] = struct.pack("<d", first + 0.000001) + parts[name][8:]
        manifest["artifacts"][name] = {
            "sha256": hashlib.sha256(parts[name]).hexdigest(), "size_bytes": len(parts[name])}
    elif fault == "mapping_correspondence":
        mapping["atoms"][0]["source_atom_id"] = str(uuid4())
    elif fault == "particle_correspondence":
        manifest["particles"][0]["topology_atom_id"] = manifest["atoms"][1]["atom_id"]
    elif fault == "source_bytes":
        parts[manifest["sources"][0]["artifact"]] += b"\n# changed source\n"
    elif fault == "unsupported_xml":
        parts["system.xml"] = parts["system.xml"].replace(b'type="System"', b'type="UnknownPlugin"')
        manifest["artifacts"]["system.xml"] = {
            "sha256": hashlib.sha256(parts["system.xml"]).hexdigest(), "size_bytes": len(parts["system.xml"])}
    elif fault not in ("duplicate_manifest_key", "duplicate_mapping_key"):
        raise AssertionError(f"unknown rejection fixture: {fault}")
    if fault in {"source_bonds_removed", "source_residue_name_changed", "source_polymer_chain_changed"}:
        mapping["sources"] = [{key: source[key] for key in (
            "source_id", "content_sha256", "atoms", "residue_bindings", "bonds", "polymer_sequences")}
            for source in manifest["sources"]]
    parts["mapping.json"] = json.dumps(mapping, allow_nan=False).encode()
    if fault == "duplicate_mapping_key":
        parts["mapping.json"] = b'{"schema_version": 2,' + parts["mapping.json"][1:]
    # Keep mutations hash-consistent so source contradictions reach the strict
    # load-only semantic verifier rather than failing an artifact digest check.
    manifest["artifacts"]["mapping.json"] = {
        "sha256": hashlib.sha256(parts["mapping.json"]).hexdigest(), "size_bytes": len(parts["mapping.json"])}
    parts["manifest.json"] = json.dumps(manifest, allow_nan=False).encode()
    if fault == "duplicate_manifest_key":
        parts["manifest.json"] = b'{"schema_version": 2,' + parts["manifest.json"][1:]
    return parts


@pytest.mark.parametrize("fault", [
    "unknown_manifest_version", "boolean_manifest_version", "floating_manifest_version",
    "unknown_mapping_version", "boolean_mapping_version", "floating_mapping_version",
    "mixed_versions", "missing_source_bonds", "missing_source_residue_bindings",
    "missing_source_polymer_sequences", "source_bonds_removed", "source_residue_name_changed",
    "source_polymer_chain_changed", "relabelled_source", "extra_load_step", "translation_history",
    "moved_f64_coordinate",
    "mapping_correspondence", "particle_correspondence", "source_bytes", "unsupported_xml",
    "duplicate_manifest_key", "duplicate_mapping_key",
])
def test_v2_rejection_never_registers_assets_or_discards_upload(database_dsn, tmp_path, fault):
    async def scenario():
        parts, scope = rejected_parts(fault, tmp_path), str(uuid4())
        async with store(database_dsn, tmp_path) as inputs:
            uploaded = await upload(inputs, scope, parts)
            for _ in range(2):  # Same request identity replays its durable failure.
                with pytest.raises(LedgerError, match="integrity_mismatch"):
                    await commit_bundle(inputs, scope, uploaded["upload_id"])
            for read in (inputs.prepared, inputs.topology):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(scope, uploaded["upload_id"])
            for part in uploaded["parts"]:
                original = tmp_path / "uploads" / uploaded["upload_id"] / part["part_id"]
                assert original.read_bytes() == parts[part["descriptor"]["name"]]

    asyncio.run(scenario())


def test_v2_repair_history_remains_outside_initial_load_only_intake(database_dsn, tmp_path):
    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope = str(uuid4())
            uploaded = await upload(inputs, scope, repair_fixture_bytes())
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, uploaded["upload_id"])
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, uploaded["upload_id"])

    asyncio.run(scenario())


def test_v1_mapping_numeric_version_preserves_existing_schema_behavior(database_dsn, tmp_path):
    async def scenario():
        parts, scope = fixture_bytes(1, tmp_path), str(uuid4())
        manifest, mapping = (json.loads(parts[name]) for name in ("manifest.json", "mapping.json"))
        mapping["schema_version"] = 1.0
        parts["mapping.json"] = json.dumps(mapping, allow_nan=False).encode()
        manifest["artifacts"]["mapping.json"] = {
            "sha256": hashlib.sha256(parts["mapping.json"]).hexdigest(),
            "size_bytes": len(parts["mapping.json"])}
        parts["manifest.json"] = json.dumps(manifest, allow_nan=False).encode()
        async with store(database_dsn, tmp_path) as inputs:
            uploaded = await upload(inputs, scope, parts)
            assert (await commit_bundle(inputs, scope, uploaded["upload_id"]))["state"] == "validated"

    asyncio.run(scenario())


def mixed_version_verifier(job, limits, control, parent_pid, raw, deadline):
    """Fault at the real child handoff after a valid v2 page was acknowledged."""
    from neomd_server.assets import TopologySender, _verify_bundle_impl
    from neomd_server.processes import arm_parent_death

    arm_parent_death(parent_pid)

    class Sender(TopologySender):
        def _send(self, prepared, version, document, data):
            if self.sequence == 1:
                Path(job["root"], "version-switch-reached").touch()
                version = 1
            super()._send(prepared, version, document, data)

    try:
        result = _verify_bundle_impl(job, limits, sender=Sender(control, raw, job["fence"], deadline))
        control.put(result)
        control.flush_until(deadline)
    except Exception:
        try:
            control.put({"error": "integrity_mismatch"})
            control.flush_until(deadline)
        except Exception:
            pass
    finally:
        control.close()
        raw.close()


def test_v2_topology_handoff_pins_first_page_version(database_dsn, tmp_path, monkeypatch):
    from neomd_server import assets

    monkeypatch.setattr(assets, "_verify_bundle_process", mixed_version_verifier)

    async def scenario():
        async with store(database_dsn, tmp_path) as inputs:
            scope, parts = str(uuid4()), load_fixture_bytes(tmp_path)
            uploaded = await upload(inputs, scope, parts)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, uploaded["upload_id"])
            assert (tmp_path / "version-switch-reached").exists()
            for read in (inputs.prepared, inputs.topology):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(scope, uploaded["upload_id"])

    asyncio.run(scenario())


@pytest.mark.parametrize("relabelled", [False, True])
def test_new_grouping_schema_does_not_expand_load_only_service_intake(database_dsn, tmp_path, relabelled):
    from neomd.prepare import verify_prepared_system
    from neomd_server.prepared_validation import validate_prepared
    from tests.v2.test_protein_grouping import export_grouped

    fixture_root = tmp_path / "group-fixture"
    fixture_root.mkdir()
    path, _, _ = export_grouped(fixture_root)
    manifest = verify_prepared_system(path, profile="canonical_protein_group_v1")
    validate_prepared("manifest", manifest)  # New shape is supported, service scope is not.
    parts = {name: (path.parent / name).read_bytes()
             for name in ("manifest.json", *manifest["artifacts"])}
    if relabelled:
        manifest["preparation_steps"] = manifest["preparation_steps"][:1]
        parts["manifest.json"] = json.dumps(manifest).encode()

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as inputs:
            uploaded = await upload(inputs, scope, parts)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, uploaded["upload_id"])
            with pytest.raises(LedgerError, match="not_found"):
                await inputs.prepared(scope, uploaded["upload_id"])
            assert (tmp_path / "uploads" / uploaded["upload_id"]).is_dir()

    asyncio.run(scenario())


@pytest.mark.parametrize("relabelled", [False, True])
def test_real_grouped_heavy_repair_does_not_expand_load_only_service_intake(database_dsn, tmp_path, relabelled):
    from neomd.assembly_report import assemble_protein_system, verify_protein_system_assembly
    from neomd.grouped_repair import group_and_repair_protein
    from neomd.prepared import file_digest, read_prepared_bundle
    from tests.v2.test_grouped_repair import complete_source
    from tests.v2.test_system_assembly_report import PARAMETERS

    fixture = tmp_path / "repair-fixture"
    fixture.mkdir()
    repaired = group_and_repair_protein(complete_source(fixture), platform="CPU", seed=123)
    bundle = assemble_protein_system(repaired.topology, repaired.positions_nm, fixture / "assembled",
                                     lineage=repaired.lineage, parameters=PARAMETERS)
    assembled = Path(bundle.prepared_manifest)
    verify_protein_system_assembly(assembled, expected_sha256=file_digest(assembled))
    actual = read_prepared_bundle(assembled)
    # The wire upload vocabulary does not admit standalone reports/resources.
    # Publish the same actual System and observed history without those optional
    # artifacts to exercise the later LOAD-only gate, not an earlier name error.
    path = Path(repaired.lineage.publish(
        fixture / "service-export", topology=repaired.topology, positions_nm=repaired.positions_nm,
        particles=actual["particles"], coordinates_nm=repaired.positions_nm,
        box_vectors_nm=actual["box_vectors_nm"], parameters=actual["parameters"],
        artifacts={"topology.pdbx": assembled.parent / "topology.pdbx",
                   "system.xml": assembled.parent / "system.xml"},
    ))
    manifest = read_prepared_bundle(path)
    assert [row["operation"] for row in manifest["preparation_steps"]] == [
        "load", "group_protein_water", "repair_standard_residue"]
    parts = {name: (path.parent / name).read_bytes()
             for name in ("manifest.json", *manifest["artifacts"])}
    if relabelled:
        manifest["preparation_steps"] = manifest["preparation_steps"][:1]
        parts["manifest.json"] = json.dumps(manifest).encode()

    async def scenario():
        scope = str(uuid4())
        async with store(database_dsn, tmp_path) as inputs:
            uploaded = await upload(inputs, scope, parts)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, uploaded["upload_id"])
            for read in (inputs.prepared, inputs.topology):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(scope, uploaded["upload_id"])
            assert (tmp_path / "uploads" / uploaded["upload_id"]).is_dir()

    asyncio.run(scenario())
