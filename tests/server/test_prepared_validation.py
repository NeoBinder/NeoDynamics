"""Public prepared-validator parity with frozen v1/v2 schemas and JCS."""

import copy
import json
from pathlib import Path

import pytest

from neomd_server.contract import ContractError, canonical_bytes, validate
from neomd_server.prepared_validation import validate_prepared

CONTRACTS = Path(__file__).parents[2] / "contracts/prepared-bundle"
FIXTURES = {1: "water-extra-site", 2: "standard-heavy-repair"}


def document(version, kind):
    return json.loads((CONTRACTS / f"v{version}/fixtures/{FIXTURES[version]}/{kind}.json").read_text())


def outcome(function, *args):
    try:
        function(*args)
    except ContractError as error:
        return error.problems
    return ()


def parity(version, kind, value, *, valid):
    before = copy.deepcopy(value)
    expected = outcome(validate, f"urn:neodyn:prepared-bundle:v{version}:{kind}", value)
    actual = outcome(validate_prepared, kind, value)
    assert actual == expected
    assert bool(actual) is not valid
    assert value == before


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kind", ["manifest", "mapping"])
def test_exported_fixtures_and_numeric_constants_keep_exact_jcs(version, kind):
    value = document(version, kind)
    raw = canonical_bytes(value)
    assert validate_prepared(kind, value) is value
    assert canonical_bytes(value) == raw
    parity(version, kind, value, valid=True)
    value["schema_version"] = float(version)
    value["topology_particle_indices"] = [float(index) for index in value["topology_particle_indices"]]
    parity(version, kind, value, valid=True)
    assert canonical_bytes(value) == raw


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kind", ["manifest", "mapping"])
@pytest.mark.parametrize("case", ["unknown", "missing", "uuid", "boolean_index", "boolean_control",
                                  "duplicate_index", "null_atom", "extra_source", "parent_uuid",
                                  "unsafe_integer", "unicode", "nonfinite"])
def test_negative_corpus_matches_all_reference_problems(version, kind, case):
    value = document(version, kind)
    if case == "unknown":
        value["unexpected"] = "not accepted"
    elif case == "missing":
        del value["atoms"][0]["atom_id"]
        del value["particles"][0]["identity_id"]
    elif case == "uuid":
        value["prepared_revision_id"] = "not-a-uuid"
        value["particles"][0]["identity_id"] = "A" * 36
    elif case == "boolean_index":
        value["particles"][0]["particle_index"] = True
    elif case == "boolean_control":
        value["particles"][0]["control_eligible"] = 1
    elif case == "duplicate_index":
        value["topology_particle_indices"] = [0, 0.0]
    elif case == "null_atom":
        value["atoms"][0] = None
    elif case == "extra_source":
        value["sources"][0]["atoms"][0]["unexpected"] = True
    elif case == "parent_uuid":
        value["atoms"][0]["source_atom_id"] = 17
    elif case == "unsafe_integer":
        value["particles"][0]["particle_index"] = 2**53
    elif case == "unicode":
        value["atoms"][0]["atom_id"] = "\ud800"
    elif case == "nonfinite":
        value["sources"][0]["atoms"][0]["position_nm"][0] = float("inf")
    parity(version, kind, value, valid=False)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("case", ["mass", "shape", "unit", "virtual_site", "bond", "enum",
                                  "hash", "boolean_size", "box"])
def test_manifest_specific_constraints_are_not_replaced_by_mapping(version, case):
    value = document(version, "manifest")
    if case == "mass":
        value["particles"][0]["mass_dalton"] = -1
    elif case == "shape":
        value["coordinates"]["shape"] = [1, 3, 4]
    elif case == "unit":
        value["coordinates"]["unit"] = "angstrom"
    elif case == "virtual_site":
        value["particles"][0]["virtual_site"] = {"type": "LocalCoordinatesSite"}
    elif case == "bond":
        value["bonds"] = [{"atom_ids": [value["atoms"][0]["atom_id"]] * 2, "order": 1, "type": None}]
    elif case == "enum":
        value["preparation_steps"] = [{"operation": "unknown"}]
    elif case == "hash":
        value["artifacts"]["system.xml"]["sha256"] = "A" * 64
    elif case == "boolean_size":
        value["artifacts"]["system.xml"]["size_bytes"] = True
    elif case == "box":
        value["box_vectors_nm"] = [[1, 2, 3], [4, 5, 6]]
    parity(version, "manifest", value, valid=False)


@pytest.mark.parametrize("kind", ["manifest", "mapping"])
@pytest.mark.parametrize("case", ["source_bond", "unknown_polymer", "known_polymer", "repair"])
def test_v2_lineage_and_conditional_constraints(kind, case):
    value = document(2, kind)
    source = value["sources"][0]
    if case == "source_bond":
        source["bonds"][0]["aromatic"] = 0
    elif case == "unknown_polymer":
        source["polymer_sequences"][0]["evidence"] = "unknown"
        source["polymer_sequences"][0]["residue_names"] = ["ALA"]
    elif case == "known_polymer":
        source["polymer_sequences"][0]["evidence"] = "client_declared"
        source["polymer_sequences"][0]["residue_names"] = None
    elif case == "repair":
        if kind == "mapping":
            source["atoms"][0]["formal_charge"] = True
        else:
            value["preparation_steps"] = [{"operation": "repair_standard_residue", "atom_count": 1,
                                           "generated_count": 1, "repair": {"tool": {"name": "other"}}}]
    parity(2, kind, value, valid=False)


@pytest.mark.parametrize("kind", ["manifest", "mapping"])
def test_bool_version_never_passes_as_one_and_unknown_versions_fail_closed(kind):
    value = document(1, kind)
    value["schema_version"] = True
    parity(1, kind, value, valid=False)
    for version in (None, False, 0, 3, "1", [1], {"version": 1}):
        value["schema_version"] = version
        with pytest.raises(ContractError):
            validate_prepared(kind, value)
    with pytest.raises(ValueError, match="unknown prepared schema kind"):
        validate_prepared("source-input", value)


@pytest.mark.parametrize("kind", ["manifest", "mapping"])
def test_schema_validation_does_not_invent_correspondence_or_authenticity(kind):
    value = document(1, kind)
    # Cross-record identity correspondence belongs to the existing scientific
    # reader, not schema validation or proof of origin.
    value["atoms"][1]["atom_id"] = value["atoms"][0]["atom_id"]
    parity(1, kind, value, valid=True)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("site_type", ["TwoParticleAverageSite", "ThreeParticleAverageSite",
                                       "OutOfPlaneSite", "LocalCoordinatesSite"])
def test_valid_virtual_site_forms_keep_draft_behavior(version, site_type):
    value = document(version, "manifest")
    ids = [atom["atom_id"] for atom in value["atoms"][:3]]
    site = {"type": site_type, "parent_indices": list(range(len(ids))), "parent_ids": ids}
    if site_type == "LocalCoordinatesSite":
        site.update(origin_weights=[1, 0, 0], x_weights=[-1, 1, 0], y_weights=[-1, 0, 1],
                    local_position_nm=[0.1, 0.2, 0.3])
    else:
        site["weights"] = [0.5, 0.5]
        if site_type == "OutOfPlaneSite":
            site["cross_weight_unit"] = "1/nm"
    value["particles"][0]["virtual_site"] = site
    parity(version, "manifest", value, valid=True)


def test_unknown_schema_version_still_collects_jcs_problems():
    value = document(1, "manifest")
    value["schema_version"] = 7
    value["parameters"]["invalid"] = float("nan")
    with pytest.raises(ContractError) as error:
        validate_prepared("manifest", value)
    assert {path for path, _ in error.value.problems} == {"/schema_version", "/"}
