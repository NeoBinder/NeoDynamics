"""TorchScript export of a trained linear ML-CV model (ADR-0006).

The ``.pt`` is the phase-2 handoff artifact (docs/adr/0006-mlcv-injection-torchcv.md):
``act((x - mean) @ W^T + b)`` over the FEATURE vector, pinned bit-tightly to
:func:`neomd.mlcv.apply_model` (:func:`convert`).  torch is imported lazily —
a missing torch is a clean user error, exit 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from neomd.errors import NeoUserError

from .models import MODEL_TYPES, load_model

__all__ = ["convert", "ConvertResult"]

_TORCH_MESSAGE = (
    "torch is not installed — the TorchScript export (`neomd mlcv convert`) "
    "needs torch in this environment (any interpreter with torch works; the "
    "numpy-side model artifact stays usable without it)")


@dataclass(frozen=True)
class ConvertResult:
    """What ``convert`` did (the CLI summary is built from this)."""

    output: str
    model_type: str
    n_outputs: int


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise NeoUserError(_TORCH_MESSAGE) from error
    return torch


def convert(model_path, output: str | None = None) -> ConvertResult:
    """model.npz -> TorchScript module (``torch.jit.script`` of the exact
    linear weights, mean folded into the bias); the saved ``.pt``
    round-trips through ``torch.jit.load`` and reproduces
    :func:`neomd.mlcv.apply_model` bit-tightly."""
    torch = _require_torch()

    header, arrays = load_model(model_path)
    model_type = header.get("model_type")
    if model_type not in MODEL_TYPES:
        from .models import _unknown_model_error

        raise _unknown_model_error(model_type)

    class LinearCV(torch.nn.Module):
        """The phase-1 export shape: float64 linear map + activation.

        Constants (``__constants__``) are baked at script time; the feature
        names and format version ride along as plain string attributes for
        provenance (they do not enter ``forward``).
        """

        __constants__ = ("sigmoid", "format_version", "feature_names",
                         "model_type")

        def __init__(self, weight, bias, sigmoid, feature_names):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
            self.sigmoid = bool(sigmoid)
            self.format_version = 1
            self.feature_names = ";".join(str(n) for n in feature_names)
            self.model_type = str(model_type)

        def forward(self, x):
            y = torch.matmul(x, self.weight.t()) + self.bias
            if self.sigmoid:
                return torch.sigmoid(y)
            return y

    if model_type == "tica":
        components = arrays["components"]  # (k, d)
        mean = arrays["mean"]  # (d,)
        weight = components
        bias = -(mean @ components.T)  # the TICA mean, folded into the bias
    else:  # logistic — weights/bias are already in raw feature units
        weight = arrays["weights"][None, :]  # (1, d)
        bias = arrays["bias"]  # (1,)
    module = LinearCV(
        weight=torch.as_tensor(np.asarray(weight), dtype=torch.float64),
        bias=torch.as_tensor(np.asarray(bias), dtype=torch.float64),
        sigmoid=(model_type == "logistic"),
        feature_names=header.get("feature_names", []))
    module.eval()
    scripted = torch.jit.script(module)
    out = str(output or os.path.splitext(os.fspath(model_path))[0] + ".pt")
    scripted.save(out)
    return ConvertResult(output=out, model_type=str(model_type),
                         n_outputs=int(np.asarray(weight).shape[0]))
