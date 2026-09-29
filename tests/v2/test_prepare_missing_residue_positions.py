"""Public preparation rejects polymers with known unpositioned residues."""

import json

import pytest

from neomd.cli import main
from neomd.errors import ConfigValueError
from neomd.system import PlainForceFieldBuilder, prepare_system


def test_prepare_system_rejects_pdb_seqres_residue_without_coordinates(tmp_path):
    source = tmp_path / "missing.pdb"
    source.write_text(
        "SEQRES   1 A    2  ALA GLY\n"
        "ATOM      1  N   ALA A   1      11.104  13.207   8.678  1.00 20.00           N\n"
        "END\n"
    )
    output = tmp_path / "prepared"

    with pytest.raises(ConfigValueError) as caught:
        prepare_system(
            {"protein": {"path": str(source)}, "output_dir": str(output)},
            forcefield=PlainForceFieldBuilder(),
        )

    assert caught.value.key == "protein"
    assert caught.value.source == str(source)
    assert "2 SEQRES residues but has 1 matching coordinate residues" in caught.value.message
    assert "upstream modeling" in caught.value.message
    assert not output.exists()


def test_cli_rejects_pdbx_declared_residue_without_coordinates(tmp_path, capsys):
    source = tmp_path / "missing.pdbx"
    source.write_text(
        "data_missing\n"
        "loop_\n"
        "_entity_poly_seq.entity_id\n"
        "_entity_poly_seq.num\n"
        "_entity_poly_seq.mon_id\n"
        "1 1 ALA\n"
        "1 2 GLY\n"
        "loop_\n"
        "_struct_asym.id\n"
        "_struct_asym.entity_id\n"
        "A 1\n"
        "loop_\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_entity_id\n"
        "_atom_site.label_seq_id\n"
        "A 1 1\n"
    )
    output = tmp_path / "prepared"
    config = tmp_path / "prepare.json"
    config.write_text(json.dumps({"protein": {"path": str(source)},
                                  "output_dir": str(output)}))

    assert main(["prepare", str(config)]) == 2
    captured = capsys.readouterr()
    assert "1 declared residues without coordinate positions" in captured.err
    assert "sequence positions: 2" in captured.err
    assert "model missing residues upstream" in captured.err
    assert "prepared system:" not in captured.out
    assert not output.exists()
