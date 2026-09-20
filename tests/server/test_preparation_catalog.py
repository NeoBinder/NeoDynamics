"""Accepted preset resources bind exact installed bytes without assembly defaults."""

import hashlib
import importlib.resources
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from neomd_server.contract import validate
from neomd_server.preparation import preparation_catalog


def test_catalog_matches_exact_resource_bytes_and_declares_incomplete_assembly():
    catalog = preparation_catalog()
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "contracts/protein-preparation/v1/catalog.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(catalog)
    validate("urn:neodyn:protein-preparation:v1:catalog", catalog)
    preset = catalog["presets"][0]
    assert preset["editable_parameters"] == ["ionic_strength_mol_per_l", "padding_nm", "ph"]
    for resource in preset["resources"]:
        data = (importlib.resources.files("openmmforcefields") / "ffxml" / resource["resource_id"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == resource["sha256"]
        assert len(data) == resource["size_bytes"]
    material = {key: value for key, value in preset.items() if key != "preset_revision_sha256"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == preset["preset_revision_sha256"]
    assert catalog["system_assembly_available"] is True
    assert catalog["limits"] == {
        "max_source_bytes": 67108864,
        "max_result_bytes": 536870912,
        "time_limit_seconds": 3600,
        "memory_limit_bytes": 4294967296,
        "max_concurrent_operations": 1,
        "max_queued_per_owner": 8,
    }
    assert catalog["unresolved_parameters"] == []
    assert "hydrogen_mass" not in catalog["unresolved_parameters"]
    assert catalog["presets"][0]["accepted_parameters"]["hydrogen_mass_amu"] == 4.0
    assert "existing_water" not in catalog["unresolved_parameters"]
    assert catalog["presets"][0]["accepted_parameters"]["existing_water"] == "retain_and_solvate"
    assert "constraints" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["constraints"] == "HBonds"
    assert "rigid_water" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["rigid_water"] is True
    assert preset["accepted_parameters"]["remove_cm_motion"] is False
    assert "remove_cm_motion" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["nonbonded_method"] == "pme"
    assert "nonbonded_method" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["nonbonded_cutoff_nm"] == 1.0
    assert "nonbonded_cutoff_nm" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["ewald_error_tolerance"] == 0.0005
    assert "ewald_error_tolerance" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["use_dispersion_correction"] is True
    assert "use_dispersion_correction" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["switch_distance_nm"] is None
    assert "switch_distance_nm" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["exceptions_use_periodic_boundary_conditions"] is False
    assert "exceptions_use_periodic_boundary_conditions" not in catalog["unresolved_parameters"]
    assert preset["accepted_parameters"]["flexible_constraints"] is False
