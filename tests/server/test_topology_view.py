"""Exact derived topology bytes and bounded transport, without simulation claims."""

import copy
import hashlib
import json
import tracemalloc
from pathlib import Path
from uuid import UUID

import pytest

from neomd_server.contract import canonical_bytes, content_hash, validate
from neomd_server.topology import (
    COLLECTIONS,
    DESCRIPTOR_BYTES,
    PAGE_BYTES,
    TopologyLimitError,
    parse_topology_bytes,
    project_topology,
)

ROOT = Path(__file__).resolve().parents[2]


def fixture(version=1):
    name = "water-extra-site" if version == 1 else "standard-heavy-repair"
    raw = (ROOT / f"contracts/prepared-bundle/v{version}/fixtures/{name}/manifest.json").read_bytes()
    manifest = json.loads(raw)
    prepared = {"bundle_id": str(UUID(int=101)),
                "prepared_revision_id": manifest["prepared_revision_id"],
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "mapping_sha256": manifest["artifacts"]["mapping.json"]["sha256"]}
    return manifest, prepared


def collect(manifest, prepared):
    pages = []
    descriptor = project_topology(manifest, prepared, lambda *row: pages.append(row))
    return descriptor, pages


def check_projection(manifest, prepared, raw, pages):
    view = parse_topology_bytes(raw, limit=DESCRIPTOR_BYTES)
    validate("urn:neodyn:topology-view:v1:view", view)
    assert content_hash({k: v for k, v in view.items() if k != "view_sha256"}) == view["view_sha256"]
    assert view["prepared"] == prepared
    assert view["prepared_schema_version"] == manifest["schema_version"]
    assert view["particle_count"] == manifest["particle_count"]
    for collection in view["collections"]:
        selected = [p for p in pages if p[0] == collection["kind"]]
        reconstructed = []
        for spec, (kind, index, first, count, data) in zip(collection["pages"], selected, strict=True):
            page = parse_topology_bytes(data, limit=PAGE_BYTES)
            validate("urn:neodyn:topology-view:v1:page", page)
            assert spec == {"page_index": index, "first_row": first, "row_count": count,
                            "size_bytes": len(data), "content_sha256": hashlib.sha256(data).hexdigest()}
            assert page["prepared"] == prepared and page["kind"] == kind
            assert first == len(reconstructed) and count == len(page["rows"])
            reconstructed.extend(page["rows"])
        assert reconstructed == manifest[collection["kind"]]
        assert len(reconstructed) == collection["row_count"]
    return view


@pytest.mark.parametrize("version", [1, 2])
def test_projection_preserves_exact_source_tables_nulls_sites_and_particle_order(version):
    manifest, prepared = fixture(version)
    raw, pages = collect(manifest, prepared)
    check_projection(manifest, prepared, raw, pages)
    assert collect(manifest, prepared) == (raw, pages)


def test_byte_pagination_unicode_and_512_row_pagination_are_exact():
    manifest, prepared = fixture()
    # Synthetic transport rows only. No new prepared System is asserted here.
    row = manifest["atoms"][0]
    manifest["atoms"] = [{**row, "atom_id": str(UUID(int=1000 + i)), "name": "原子🧬" * 900}
                         for i in range(700)]
    manifest["topology_particle_indices"] = list(range(1100))
    raw, pages = collect(manifest, prepared)
    check_projection(manifest, prepared, raw, pages)
    atom_pages = [p for p in pages if p[0] == "atoms"]
    assert len(atom_pages) > 2 and all(p[3] < 512 for p in atom_pages)
    assert [p[3] for p in pages if p[0] == "topology_particle_indices"] == [512, 512, 76]


def test_oversized_row_and_aggregate_limits_never_return_a_partial_descriptor(monkeypatch):
    manifest, prepared = fixture()
    too_large = copy.deepcopy(manifest)
    too_large["atoms"][0]["name"] = "x" * PAGE_BYTES
    emitted = []
    with pytest.raises(TopologyLimitError, match="one topology row"):
        project_topology(too_large, prepared, lambda *row: emitted.append(row))
    assert not emitted
    for name, limit in [("MAX_PAGES", 1), ("MAX_VIEW_BYTES", 1), ("DESCRIPTOR_BYTES", 1)]:
        with monkeypatch.context() as patch:
            patch.setattr(f"neomd_server.topology.{name}", limit)
            with pytest.raises(TopologyLimitError):
                collect(manifest, prepared)


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x": {"y":1,"y":2}}',
                                  b'{} {}', b'{"x":NaN}', b'{"x":Infinity}', b'{}\n',
                                  b'{"x":9007199254740992}', b'{"x":"\\ud800"}'])
def test_untrusted_topology_bytes_are_strict_before_schema_parsing(data):
    with pytest.raises((ValueError, UnicodeError)):
        parse_topology_bytes(data, limit=PAGE_BYTES)


def test_changed_prepared_binding_and_same_count_swapped_rows_change_view_identity():
    manifest, prepared = fixture()
    before, _ = collect(manifest, prepared)
    changed = copy.deepcopy(manifest)
    changed["atoms"].reverse()
    after, _ = collect(changed, prepared)
    assert json.loads(before)["view_sha256"] != json.loads(after)["view_sha256"]
    other, _ = collect(manifest, {**prepared, "bundle_id": str(UUID(int=102))})
    assert json.loads(before)["view_sha256"] != json.loads(other)["view_sha256"]
    with pytest.raises(ValueError, match="revision"):
        collect(manifest, {**prepared, "prepared_revision_id": str(UUID(int=102))})
    with pytest.raises(ValueError, match="mapping"):
        collect(manifest, {**prepared, "mapping_sha256": "0" * 64})


def test_oversized_row_rejects_without_allocating_the_whole_encoded_value():
    manifest, prepared = fixture()
    for text in ["x" * (16 * PAGE_BYTES), "\u0000" * (PAGE_BYTES // 2)]:
        manifest["atoms"][0]["name"] = text
        tracemalloc.start()
        try:
            with pytest.raises(TopologyLimitError):
                collect(manifest, prepared)
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 2 * PAGE_BYTES
        finally:
            tracemalloc.stop()


def test_frozen_golden_pages_and_descriptor_are_exact():
    manifest, prepared = fixture()
    raw, pages = collect(manifest, prepared)
    golden = ROOT / "contracts/topology-view/v1/fixtures/water-extra-site"
    assert (golden / "view.json").read_bytes() == raw
    for kind, index, _, _, data in pages:
        assert (golden / f"{kind}-{index}.json").read_bytes() == data
    assert list(COLLECTIONS) == [p["kind"] for p in json.loads(raw)["collections"]]
    assert canonical_bytes(json.loads(raw)) == raw
