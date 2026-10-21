"""Public preparation must preserve a deposited covalent attachment or fail."""

import json
from pathlib import Path

import pytest
from openmm import app

from neomd.cli import main
from neomd.errors import ConfigValueError
from neomd_preparation import PlainForceFieldBuilder, prepare_system

_POSE = Path(__file__).parent / "fixtures" / "1cgh_ser195_1zg.pdb"


@pytest.mark.parametrize("with_custom_bonds", [False, True])
def test_1cgh_ser195_1zg_attachment_rejects_before_any_prepared_artifact(
    tmp_path, with_custom_bonds,
):
    # PDB 1CGH deposited SER195/1ZG coordinates and its explicit OG--P link.
    # The bounded fixture retains every atom of the two source residues.
    topology = app.PDBFile(str(_POSE)).topology
    assert topology.getNumAtoms() == 46
    assert [
        (left.residue.name, left.residue.id, left.name,
         right.residue.name, right.residue.id, right.name)
        for left, right in topology.bonds() if left.residue is not right.residue
    ] == [("SER", "195", "OG", "1ZG", "1", "P")]

    output = tmp_path / "prepared"
    protein = {"path": str(_POSE)}
    if with_custom_bonds:
        # This legacy option rebuilds bonds. The deposited edge must be
        # rejected before that rebuilding could erase it.
        protein["custom_res_bonds"] = {"1ZG": {}}
    with pytest.raises(ConfigValueError) as caught:
        prepare_system(
            {"protein": protein, "output_dir": str(output)},
            forcefield=PlainForceFieldBuilder(),
            prepared_export={},
        )
    assert caught.value.key == "protein"
    assert caught.value.source == str(_POSE)
    assert "SER A:195:OG -- 1ZG B:1:P" in caught.value.message
    assert "qualified joint parameter template" in caught.value.message
    assert not output.exists()


def test_1cgh_cli_reports_attachment_without_success_receipt(tmp_path, capsys):
    output = tmp_path / "prepared"
    config_file = tmp_path / "prepare.yaml"
    config_file.write_text(json.dumps({
        "protein": {"path": str(_POSE)}, "output_dir": str(output),
    }))

    assert main(["prepare", str(config_file)]) == 2
    captured = capsys.readouterr()
    assert "SER A:195:OG -- 1ZG B:1:P" in captured.err
    assert "qualified joint parameter template" in captured.err
    assert "prepared system:" not in captured.out
    assert not output.exists()
