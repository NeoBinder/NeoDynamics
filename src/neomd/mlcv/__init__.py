"""neomd.mlcv — ML collective variables, phase 1: ``featurize | train | convert`` (ADR-0006).

numpy-only, zero simulation-core changes; the phase-2 TorchCV injection is
designed in docs/adr/0006-mlcv-injection-torchcv.md.  User guide:
docs/methods/mlcv.md.
"""

from __future__ import annotations

from .featurize import (
    FEATURE_FORMAT_VERSION,
    FEATURE_TYPES,
    MASS_DEPENDENT_TYPES,
    FeaturizeResult,
    featurize,
    validate_featurize_config,
)
from .models import (
    MODEL_FORMAT_VERSION,
    MODEL_TYPES,
    TrainResult,
    apply_model,
    load_features,
    load_model,
    save_model,
    train,
    train_logistic,
    train_tica,
)
from .torch_export import ConvertResult, convert

__all__ = [
    "FEATURE_FORMAT_VERSION",
    "FEATURE_TYPES",
    "MASS_DEPENDENT_TYPES",
    "MODEL_FORMAT_VERSION",
    "MODEL_TYPES",
    "FeaturizeResult",
    "TrainResult",
    "ConvertResult",
    "featurize",
    "validate_featurize_config",
    "train",
    "train_tica",
    "train_logistic",
    "convert",
    "apply_model",
    "load_features",
    "load_model",
    "save_model",
]
