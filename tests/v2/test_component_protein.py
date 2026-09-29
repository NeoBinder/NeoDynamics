"""Full-complex, source-bound protein variant preflight."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from neomd import component_protein
from neomd.component_contract import canonical_sha256
from neomd.component_protein import inspect_protein_variants
from neomd.protein_hydrogens import _trusted_inventory

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "contracts/molecular-topology/v1/fixtures/cross-boundary.json"


def _bond(left, right):
    return {
        "atom_ids": sorted((left, right), key=int),
        "order": {"kind": "single", "value": None},
        "kind": {"kind": "covalent", "value": None},
        "provenance": [{"kind": "source_record", "reference": "synthetic peptide", "properties": {}}],
    }


def _peptide(selected=("NALA", "CALA"), bases=("ALA", "ALA")):
    """A peptide with all heavy atoms and stored source bonds, but no H."""
    document = json.loads(SOURCE.read_text())
    _, templates, _, _ = _trusted_inventory()
    document["chains"] = document["chains"][:1]
    document["residues"] = [
        {"id": str(i), "chain_id": "1", "name": base, "entity_kind": "polymer",
         "source": None, "properties": {}}
        for i, base in enumerate(bases, 1)
    ]
    document["atoms"] = []
    document["bonds"] = []
    ids = {}
    for residue_id, variant in enumerate(selected, 1):
        template = templates[variant]
        for name, element in template["atoms"].items():
            if element == "H":
                continue
            atom_id = str(len(document["atoms"]) + 1)
            ids[residue_id, name] = atom_id
            document["atoms"].append({
                "id": atom_id, "residue_id": str(residue_id), "name": name,
                "element": element, "formal_charge": None, "isotope": None,
                "aromatic": None, "stereochemistry": None, "occupancy": None,
                "b_factor": None, "source": None, "properties": {},
            })
        for pair in template["bonds"]:
            if all((residue_id, name) in ids for name in pair):
                left, right = (ids[residue_id, name] for name in pair)
                document["bonds"].append(_bond(left, right))
    for residue_id in range(1, len(selected)):
        document["bonds"].append(_bond(ids[residue_id, "C"], ids[residue_id + 1, "N"]))
    document["relations"] = []
    document["source_mappings"] = []
    document["frames"] = [{
        "id": "33333333-3333-4333-8333-333333333333", "model": "1",
        "unit": "nm", "time_ps": None, "box": None,
        "positions": [{"atom_id": atom["id"], "xyz": [float(i), 0.0, 0.0]}
                      for i, atom in enumerate(document["atoms"])],
    }]
    return document, ids


def _choices():
    return [{"residue_id": "1", "variant": "NALA"},
            {"residue_id": "2", "variant": "CALA"}]


def _reasons(result):
    return {diagnostic.detail.split(":", 1)[0] for diagnostic in result.diagnostics}


def test_exact_explicit_assignments_inspect_full_source_without_mutation():
    document, ids = _peptide()
    before = copy.deepcopy(document)
    result = inspect_protein_variants(document, _choices())
    assert result.ready, result.diagnostics
    assert document == before
    assert result.source_revision_id == document["revision_id"]
    assert result.source_topology_sha256 == canonical_sha256(document)
    assert result.protein_resource_sha256 and result.hydrogen_definitions_sha256
    assert [residue.variant for residue in result.residues] == ["NALA", "CALA"]
    assert result.residues[0].external_bonds == ((ids[1, "C"], ids[2, "N"], "2"),)
    assert result.residues[0].missing_hydrogens
    # Ready means structural preflight; H completion still follows explicitly.
    assert all(residue.missing_hydrogens for residue in result.residues)


def test_missing_or_duplicate_choices_never_infer_terminal_variant():
    document, _ = _peptide()
    assert "variant_required" in _reasons(inspect_protein_variants(document, _choices()[:1]))
    assert "invalid_assignment" in _reasons(inspect_protein_variants(document, _choices() * 2))
    wrong = _choices()
    wrong[0]["variant"] = "ALA"
    assert "external_bond_mismatch" in _reasons(inspect_protein_variants(document, wrong))


def test_unsupported_variant_preserves_the_submitted_decision():
    document, _ = _peptide()
    choices = _choices()
    choices[0]["variant"] = "HIP"
    result = inspect_protein_variants(document, choices)
    assert result.residues[0].variant == "HIP"
    assert any(diagnostic.code == "capability_unavailable"
               and diagnostic.detail.startswith("unsupported_variant:")
               and diagnostic.path == "/assignments/0/variant"
               for diagnostic in result.diagnostics)


def test_internal_cysteine_is_not_a_c_terminal_template():
    document, _ = _peptide(("NALA", "CYS", "CALA"), ("ALA", "CYS", "ALA"))
    choices = [{"residue_id": str(i), "variant": variant}
               for i, variant in enumerate(("NALA", "CYS", "CALA"), 1)]
    result = inspect_protein_variants(document, choices)
    assert result.ready, result.diagnostics
    assert result.residues[1].missing_heavy_atoms == ()


def test_missing_heavy_atom_requires_repair_and_retained_h_conflict_blocks():
    document, ids = _peptide()
    cb = ids[1, "CB"]
    document["atoms"] = [row for row in document["atoms"] if row["id"] != cb]
    document["bonds"] = [row for row in document["bonds"] if cb not in row["atom_ids"]]
    document["frames"][0]["positions"] = [row for row in document["frames"][0]["positions"]
                                          if row["atom_id"] != cb]
    result = inspect_protein_variants(document, _choices())
    assert "repair_required" in _reasons(result)
    assert "CB" in result.residues[0].missing_heavy_atoms

    document, ids = _peptide()
    hydrogen = copy.deepcopy(document["atoms"][0])
    hydrogen.update(id=str(len(document["atoms"]) + 1), name="HZ9", element="H")
    document["atoms"].append(hydrogen)
    document["bonds"].append(_bond(ids[1, "N"], hydrogen["id"]))
    document["frames"][0]["positions"].append({"atom_id": hydrogen["id"], "xyz": [0, 0, 0]})
    assert "protonation_conflict" in _reasons(inspect_protein_variants(document, _choices()))


def test_retained_n_terminal_h_alias_matches_pinned_loader():
    document, ids = _peptide()
    hydrogen = copy.deepcopy(document["atoms"][0])
    hydrogen.update(id=str(len(document["atoms"]) + 1), name="H", element="H")
    document["atoms"].append(hydrogen)
    document["bonds"].append(_bond(ids[1, "N"], hydrogen["id"]))
    document["frames"][0]["positions"].append({"atom_id": hydrogen["id"], "xyz": [0, 0, 0]})
    result = inspect_protein_variants(document, _choices())
    assert result.ready, result.diagnostics
    assert "H1" not in result.residues[0].missing_hydrogens


def test_protein_ligand_bond_is_an_unsupported_attachment():
    document, ids = _peptide()
    document["residues"].append({"id": "3", "chain_id": None, "name": "LIG",
                                 "entity_kind": "nonpolymer", "source": None, "properties": {}})
    atom = copy.deepcopy(document["atoms"][0])
    atom.update(id=str(len(document["atoms"]) + 1), residue_id="3", name="C1", element="C")
    document["atoms"].append(atom)
    document["bonds"].append(_bond(ids[2, "CB"], atom["id"]))
    document["frames"][0]["positions"].append({"atom_id": atom["id"], "xyz": [0, 0, 0]})
    result = inspect_protein_variants(document, _choices())
    assert "unsupported_attachment" in _reasons(result)
    assert not result.ready
    assert any(bond[1] == atom["id"] for residue in result.residues
               for bond in residue.external_bonds)


def test_peptide_link_must_have_compatible_declared_bond_kind():
    document, _ = _peptide()
    document["bonds"][-1]["kind"]["kind"] = "disulfide"
    assert "unsupported_attachment" in _reasons(inspect_protein_variants(document, _choices()))


def test_cross_boundary_fixture_keeps_protein_ligand_attachment_visible():
    document = json.loads(SOURCE.read_text())
    result = inspect_protein_variants(document, [
        {"residue_id": "1", "variant": "ALA"},
        {"residue_id": "2", "variant": "CYX"},
        {"residue_id": "3", "variant": "CYX"},
    ])
    assert "unsupported_attachment" in _reasons(result)


def test_partial_or_malformed_source_fails_closed():
    document, _ = _peptide()
    document["coverage"]["state"] = "partial"
    assert _reasons(inspect_protein_variants(document, _choices())) == {"connectivity_incomplete"}
    document["schema_version"] = 7
    assert _reasons(inspect_protein_variants(document, _choices())) == {"invalid_topology"}


def test_diagnostics_use_frozen_component_contract_codes():
    document, _ = _peptide()
    results = [inspect_protein_variants(document, _choices()[:1]),
               inspect_protein_variants(document, _choices() * 2)]
    document["coverage"]["state"] = "partial"
    results.append(inspect_protein_variants(document, _choices()))
    schema = json.loads((ROOT / "contracts/component-preparation/v1/diagnostic.schema.json").read_text())
    allowed = set(schema["properties"]["code"]["enum"])
    assert set(component_protein._DIAGNOSTIC_CODES.values()) <= allowed
    assert all(diagnostic.code in allowed and diagnostic.path.startswith("/")
               and diagnostic.detail for result in results for diagnostic in result.diagnostics)
