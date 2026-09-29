"""OpenMM-free mapped two-component periodic observables for bounded comparisons."""

from __future__ import annotations

import math

import numpy as np

from .coordinates import evaluate_coordinates
from .stats import block_average


def evaluate_component_observables(*, positions_nm, boxes_nm, steps, time_ps,
                                   groups, bonds, reference_nm, state_by_step,
                                   instance_ids, discard_frames):
    """Return two internal heavy RMSDs, one geometric distance and diagnostics.

    ``positions_nm`` contains only selected component heavy atoms, in ``groups``
    order. The caller resolves each source identity independently against its
    prepared mapping. Unit weights make the inter-component centers geometric
    and independent of hydrogen-mass repartitioning.
    """
    positions = np.asarray(positions_nm, dtype=float)
    if positions.ndim != 3 or positions.shape[1] != sum(map(len, groups)):
        raise ValueError("selected component frame shape differs from mapping")
    specs = [{"kind": "rmsd", "pbc": "minimum_image", "reference_nm": reference_nm,
              "measure_indices": group, "fit_indices": group, "align": True}
             for group in groups]
    specs.append({"kind": "distance", "pbc": "minimum_image", "groups": groups})
    coordinates = evaluate_coordinates(
        positions, np.ones(positions.shape[1], dtype=float), specs,
        boxes_nm=boxes_nm, bonds=bonds,
    )
    selected_steps = [int(v) for v in steps[discard_frames:]]
    selected_times = [float(v) for v in time_ps[discard_frames:]]
    if not selected_steps or any(step not in state_by_step
                                 or abs(state_by_step[step][0] - time) > 1e-6
                                 for step, time in zip(selected_steps, selected_times)):
        raise ValueError("state report does not cover selected DCD frames")
    values = [coordinates[discard_frames:, 0].tolist(),
              coordinates[discard_frames:, 1].tolist(),
              coordinates[discard_frames:, 2].tolist(),
              [state_by_step[s][1] for s in selected_steps],
              [state_by_step[s][2] for s in selected_steps]]
    keys = [("ligand_internal_rmsd", instance_ids[0], "nm"),
            ("ligand_internal_rmsd", instance_ids[1], "nm"),
            ("interligand_distance", None, "nm"),
            ("temperature", None, "K"), ("volume", None, "nm^3")]
    measurements = []
    for (kind, selector, unit), series in zip(keys, values):
        if not all(map(math.isfinite, series)):
            raise ValueError("nonfinite analysis series")
        within = block_average(series).error if len(series) >= 8 else None
        measurements.append({"observable": kind, "selector": selector, "unit": unit,
                             "steps": selected_steps, "time_ps": selected_times,
                             "values": series, "mean": float(np.mean(series)),
                             "within_replica_sem": within})
    return measurements
