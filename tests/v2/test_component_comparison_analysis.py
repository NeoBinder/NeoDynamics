"""Deterministic periodic selected-heavy observables in the engine analysis API."""

import numpy as np

from neomd.analysis import evaluate_component_observables


def test_periodic_internal_pose_and_geometric_centroid_are_image_invariant():
    reference = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [3.5, 0, 0], [4.5, 0, 0], [3.5, 1, 0],
    ], dtype=float)
    frames = np.asarray([reference, reference.copy()])
    frames[1, 3:] += [4, 0, 0]
    box = np.diag([4., 4., 4.])
    measurements = evaluate_component_observables(
        positions_nm=frames, boxes_nm=box, steps=np.asarray([10, 20]),
        time_ps=np.asarray([.02, .04]), groups=[[0, 1, 2], [3, 4, 5]],
        bonds=[[0, 1], [1, 2], [3, 4], [4, 5]], reference_nm=reference,
        state_by_step={10: (.02, 300., 64.), 20: (.04, 301., 64.)},
        instance_ids=["first", "second"], discard_frames=0,
    )
    assert [m["observable"] for m in measurements] == [
        "ligand_internal_rmsd", "ligand_internal_rmsd", "interligand_distance",
        "temperature", "volume"]
    assert np.allclose(measurements[0]["values"], 0, atol=1e-7)
    assert np.allclose(measurements[1]["values"], 0, atol=1e-7)
    assert np.allclose(measurements[2]["values"], .5, atol=1e-7)
    assert measurements[3]["values"] == [300., 301.]
    assert measurements[4]["unit"] == "nm^3"
    assert all(m["within_replica_sem"] is None for m in measurements)
