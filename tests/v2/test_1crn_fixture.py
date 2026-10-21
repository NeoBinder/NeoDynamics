"""Pin the real 1CRN qualification fixture (tests/data/1crn.pdbx).

The fixture emulates the desktop capture profile: official RCSB 1CRN
atom_site rows plus disulfide struct_conn, chem_comp_bond stripped (see
tests/data/1crn.SOURCE.md). These checks guard the file against accidental
edits and pin the load+descriptor evidence path on the real structure.
"""

import hashlib
from pathlib import Path

from openmm import app

from neomd_preparation.source import load_protein_source
from tests.v2.test_protein_preparation import descriptor

DATA = Path(__file__).resolve().parents[1] / "data" / "1crn.pdbx"


def test_fixture_bytes_pinned():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == (
        "d3115032db8e0a6c9c4fff1d0259b531c70a22593a6aa1309528992ad9f663e4")


def test_real_1crn_load_and_descriptor_validate():
    loaded = app.PDBxFile(str(DATA))
    topology = loaded.topology
    residues = list(topology.residues())
    bonds = list(topology.bonds())
    assert len(residues) == 46
    assert topology.getNumChains() == 1
    assert topology.getNumAtoms() == 327
    assert len(bonds) == 337
    assert all(r.name not in ("HOH", "WAT") for r in residues)
    disulfides = [b for b in bonds if b[0].name == "SG" and b[1].name == "SG"]
    assert len(disulfides) == 3
    # Every loaded bond is typeless: official chem_comp_bond was stripped.
    assert all(b.order is None and b.type is None for b in bonds)
    binding = descriptor(DATA, loaded)
    source = load_protein_source(str(DATA), binding)
    assert source is not None
