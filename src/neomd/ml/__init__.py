"""neomd.ml — the in-tree ML/MM coupling module (ADR-0004).

openmm-ml is NOT a dependency (marker-gated cross-validation reference only);
the model file is the interface.  The package imports cleanly without openmm
AND without torch (engine imports are lazy).  Reference: docs/methods/mlmm.md,
docs/adr/0004-mlmm-in-tree-coupling.md.
"""

from .selection import (
    RESIDUE_SELECTOR_RE,
    is_residue_selector,
    match_residue_selector,
    parse_residue_selector,
    resolve_residues,
)
from .spec import (
    ML_REGION_KEYS,
    MOCK_DEFAULTS,
    MODEL_KEYS,
    MODEL_TYPES,
    MLRegion,
    flatten_indices,
    flatten_selectors,
    parse_ml_region,
)

__all__ = [
    "MLRegion",
    "parse_ml_region",
    "MODEL_TYPES",
    "ML_REGION_KEYS",
    "MODEL_KEYS",
    "MOCK_DEFAULTS",
    "flatten_indices",
    "flatten_selectors",
    "RESIDUE_SELECTOR_RE",
    "is_residue_selector",
    "parse_residue_selector",
    "match_residue_selector",
    "resolve_residues",
]
