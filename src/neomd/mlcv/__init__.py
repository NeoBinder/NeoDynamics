"""neomd.mlcv — ML collective variables (ADR-0006, phase 1).

A numpy-only, out-of-tree-style tool layer: zero simulation-core changes
(the phase-2 TorchCV injection is designed in
``docs/adr/0006-mlcv-injection-torchcv.md``).  The workflow:

    featurize   run dirs / trajectory + feature config -> features.npz
                (named feature columns over per-frame positions read from
                the runs' ``output.dcd`` and masses from their system.xml,
                reusing the PUBLIC cv registry's evaluate implementations)
    train       features.npz -> model.npz
                (TICA for unlabeled streams, logistic regression for
                labeled two-basin data — both linear, both numpy)
    convert     model.npz -> TorchScript .pt
                (the phase-2 handoff artifact; torch-gated)

CLI spelling: ``neomd mlcv featurize|train|convert ...`` (cli.py maps the
flags onto the public calls below).  Config problems render through the
errors.py collect-all family (key paths + did-you-mean), exit 2.
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
