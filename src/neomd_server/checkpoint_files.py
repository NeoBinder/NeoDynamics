"""Bounded scientific sidecars retained in immutable recovery generations.

The session hashes these names, and the worker uses the same list when collecting
completed RBFE windows. Adding a sidecar does not advertise a scientific method
or change protocol admission; it preserves state for already-instantiated runs.
"""

CHECKPOINT_COMPANIONS = (
    "manifest.json",
    "hills.npz",
    "metadynamics_bias_state.json",
    "kernels.npz",
    "gamd_calibration.json",
    "gamd_preparation.json",
    "gamd_stages.json",
    "gamd_metadata.json",
    "gamd_partitions.json",
    "ligand_exchange.json",
    "ligand_exchange_events.jsonl",
)
