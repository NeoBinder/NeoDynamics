"""Full System coordinates retain the chemical topology's explicit projection."""

from pathlib import Path

import numpy as np
import pytest

from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.port import KernelFactory, KernelSpec

FIXTURE = Path(__file__).parents[2] / "contracts/prepared-bundle/v1/fixtures/water-extra-site"


def specification(**overrides):
    positions = np.fromfile(FIXTURE / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    return KernelSpec(
        **{
            "system_xml": str(FIXTURE / "system.xml"),
            "topology_file": str(FIXTURE / "topology.pdbx"),
            "platform": "cpu",
            "seed": 71,
            "initial_positions_nm": tuple(map(tuple, positions)),
            "topology_particle_indices": (0, 1, 2),
            **overrides,
        }
    )


def test_full_particle_coordinates_and_chemical_final_structure(tmp_path):
    from openmm import app, unit

    ensure_adapters()
    kernel = KernelFactory.create(specification())
    assert kernel.num_particles == 4
    assert kernel.positions().shape == (4, 3)
    expected = np.fromfile(FIXTURE / "coordinates.f64le", dtype="<f8").reshape((-1, 3))
    np.testing.assert_allclose(kernel.positions(), expected, rtol=0, atol=1e-15)
    kernel.step(2)
    path = tmp_path / "chemical.pdbx"
    kernel.write_structure(path)
    final = app.PDBxFile(str(path))
    assert final.topology.getNumAtoms() == 3
    np.testing.assert_allclose(
        final.positions.value_in_unit(unit.nanometer), kernel.positions()[:3], atol=1e-5
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_positions_nm": ((0, 0, 0),)},
        {"initial_positions_nm": ((float("nan"), 0, 0),) * 4},
        {"topology_particle_indices": (0, 1, 1)},
        {"topology_particle_indices": (0, 1, 4)},
    ],
)
def test_invalid_full_particle_binding_rejected_before_context(overrides):
    ensure_adapters()
    with pytest.raises(ValueError):
        KernelFactory.create(specification(**overrides))
