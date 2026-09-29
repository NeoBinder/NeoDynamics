"""Independent mmCIF polymer position trust for public protein preparation."""

from pathlib import Path

import pytest

from neomd_server.preparation_execution import _trusted_pdbx_sequences

SOURCE = """data_source
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 SER
1 3 GLY
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM N N ALA A 1 1 0 0 0 1
ATOM C CA ALA A 1 1 1 0 0 1
ATOM N N GLY A 1 3 2 0 0 1
"""


def descriptor():
    return {
        "atom_bindings": [
            {"input_atom_index": 0, "atom_name": "N", "element": "N",
             "position_nm": [0, 0, 0], "source_residue_id": "r1"},
            {"input_atom_index": 1, "atom_name": "CA", "element": "C",
             "position_nm": [.1, 0, 0], "source_residue_id": "r1"},
            {"input_atom_index": 2, "atom_name": "N", "element": "N",
             "position_nm": [.2, 0, 0], "source_residue_id": "r2"},
        ],
        "residue_bindings": [
            {"input_residue_index": 0, "input_chain_index": 0,
             "name": "ALA", "source_residue_id": "r1"},
            {"input_residue_index": 1, "input_chain_index": 0,
             "name": "GLY", "source_residue_id": "r2"},
        ],
    }


def source_file(tmp_path: Path, text: str):
    path = tmp_path / "source.pdbx"
    path.write_text(text)
    return path


def test_deposited_sequence_yields_missing_position_without_descriptor_inference(tmp_path):
    found = _trusted_pdbx_sequences(source_file(tmp_path, SOURCE), descriptor())
    assert found == {0: {"residue_names": ["ALA", "SER", "GLY"],
                        "residue_bindings": [
                            {"sequence_index": 0, "source_residue_id": "r1"},
                            {"sequence_index": 2, "source_residue_id": "r2"},
                        ]}}


@pytest.mark.parametrize("mutated", [
    SOURCE.replace("A 1\nloop_\n_atom_site", "A 2\nloop_\n_atom_site"),
    SOURCE.replace("GLY A 1 3 2", "GLY A 1 2 2"),
    SOURCE.replace("ATOM N N GLY", "ATOM C N GLY"),
    SOURCE.replace("ALA A 1 1 0 0 0", "ALA A 1 1 nan 0 0"),
])
def test_inconsistent_uploaded_canonical_evidence_fails_closed(tmp_path, mutated):
    with pytest.raises(ValueError):
        _trusted_pdbx_sequences(source_file(tmp_path, mutated), descriptor())


def test_unobserved_polymer_name_is_source_authority(tmp_path):
    changed = _trusted_pdbx_sequences(source_file(tmp_path, SOURCE.replace("1 2 SER", "1 2 THR")),
                                      descriptor())
    assert changed[0]["residue_names"] == ["ALA", "THR", "GLY"]


def test_nonpolymer_water_keeps_protein_sequence_authority(tmp_path):
    source = SOURCE.replace("A 1\nloop_\n_atom_site", "A 1\nW 2\nloop_\n_atom_site")
    source += "HETATM O O HOH W 2 . 3 0 0 1\n"
    binding = descriptor()
    binding["atom_bindings"].append({
        "input_atom_index": 3, "atom_name": "O", "element": "O",
        "position_nm": [.3, 0, 0], "source_residue_id": "water",
    })
    binding["residue_bindings"].append({
        "input_residue_index": 2, "input_chain_index": 1,
        "name": "HOH", "source_residue_id": "water",
    })
    assert _trusted_pdbx_sequences(source_file(tmp_path, source), binding) == {
        0: {"residue_names": ["ALA", "SER", "GLY"],
            "residue_bindings": [
                {"sequence_index": 0, "source_residue_id": "r1"},
                {"sequence_index": 2, "source_residue_id": "r2"},
            ]},
    }
    with pytest.raises(ValueError):
        _trusted_pdbx_sequences(source_file(tmp_path, source.replace("HOH W 2 .", "LIG W 2 .")), binding)


def test_descriptor_uuid_name_and_coordinate_tampering_fails_closed(tmp_path):
    path = source_file(tmp_path, SOURCE)
    for key, value in (("source_residue_id", "wrong"), ("atom_name", "wrong"),
                       ("position_nm", [2, 0, 0])):
        changed = descriptor()
        changed["atom_bindings"][0][key] = value
        with pytest.raises(ValueError):
            _trusted_pdbx_sequences(path, changed)
