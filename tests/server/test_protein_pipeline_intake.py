"""Full SDK preparation must not silently expand actual service v2 admission."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from input_helpers import commit_bundle
from test_assets import store, upload

from neomd.prepared import read_prepared_bundle
from neomd_server.ledger import LedgerError
from neomd_server.prepared_validation import validate_prepared
from tests.v2.test_protein_pipeline import prepared


@pytest.mark.parametrize("relabelled", [False, True])
def test_full_preparation_still_rejects_at_real_load_only_commit(database_dsn, tmp_path, relabelled):
    fixture = tmp_path / "scientific"
    fixture.mkdir()
    bundle = prepared(fixture)
    path = Path(bundle.prepared_manifest)
    manifest = read_prepared_bundle(path)
    validate_prepared("manifest", manifest)
    # Optional report/ffxml names are outside the current wire vocabulary.
    # Exclude them in this supported artifact-set upload so the actual spawned
    # LOAD-only gate is exercised, not an earlier artifact-name rejection.
    for name in list(manifest["artifacts"]):
        if name == "assembly-report.json" or name.startswith("forcefield/"):
            del manifest["artifacts"][name]
    manifest["parameters"].pop("assembly_report_artifact", None)
    if relabelled:
        manifest["preparation_steps"] = manifest["preparation_steps"][:1]
    parts = {name: (path.parent / name).read_bytes() for name in manifest["artifacts"]}
    parts["manifest.json"] = json.dumps(manifest).encode()

    async def scenario():
        scope = str(uuid4())
        # Explicit test-only limits for the real solvated System, not product
        # defaults or a relaxation of the canonical-LOAD admission profile.
        async with store(database_dsn, tmp_path, capacity=20_000_000) as inputs:
            inputs.limits = replace(inputs.limits, max_xml_bytes=10_000_000, max_particles=10_000)
            uploaded = await upload(inputs, scope, parts)
            with pytest.raises(LedgerError, match="integrity_mismatch"):
                await commit_bundle(inputs, scope, uploaded["upload_id"])
            for read in (inputs.prepared, inputs.topology):
                with pytest.raises(LedgerError, match="not_found"):
                    await read(scope, uploaded["upload_id"])
            assert (tmp_path / "uploads" / uploaded["upload_id"]).is_dir()

    asyncio.run(scenario())
