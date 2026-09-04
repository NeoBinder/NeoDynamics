"""ML-CV model families + the versioned model artifact (ADR-0006, numpy-only).

Both phase-1 families LINEAR: TICA (generalized eigenproblem; lagged pairs
pooled WITHOUT crossing run boundaries — train_tica) and logistic regression
(standardization folded into the stored weights).  apply_model is the one
evaluation route, pinned by the TorchScript export.  Reference: docs/methods/mlcv.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from neomd.errors import ConfigKeyError, ConfigValueError, suggest

__all__ = [
    "MODEL_FORMAT_VERSION",
    "MODEL_TYPES",
    "TrainResult",
    "save_model",
    "load_model",
    "train",
    "train_tica",
    "train_logistic",
    "apply_model",
    "load_features",
]

#: model npz format version (bump on any layout change)
MODEL_FORMAT_VERSION = 1

MODEL_TYPES = ("tica", "logistic")


@dataclass(frozen=True)
class TrainResult:
    """What ``train`` did (the CLI summary is built from this)."""

    output: str
    model_type: str
    feature_names: list[str]
    n_frames: int
    diagnostics: dict


# ---------------------------------------------------------------------------
# the features cache (featurize.py's npz) — the train-side reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """A features.npz cache read back."""

    values: np.ndarray  # (n, d) float64
    steps: np.ndarray  # (n,) int64
    run_index: np.ndarray  # (n,) int64
    feature_names: list[str]


def load_features(path) -> FeatureSet:
    """Read a features.npz written by :func:`neomd.mlcv.featurize`."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise ConfigValueError(
            f"features file not found: {path} (run `neomd mlcv featurize` "
            f"first)", value=path)
    try:
        loaded = np.load(path, allow_pickle=False)
        if not hasattr(loaded, "files"):  # a bare .npy gives an ndarray
            raise ConfigValueError(
                f"{path} is not a features.npz cache (a bare array file — "
                f"expected an npz archive)", value=path)
        with loaded as data:
            keys = set(data.files)
            needed = {"format_version", "values", "steps", "run_index",
                      "feature_names"}
            missing = sorted(needed - keys)
            if missing:
                raise ConfigValueError(
                    f"{path} is not a features.npz cache (missing keys: "
                    f"{missing})", value=path)
            version = int(data["format_version"])
            values = np.asarray(data["values"], dtype=np.float64)
            steps = np.asarray(data["steps"], dtype=np.int64)
            run_index = np.asarray(data["run_index"], dtype=np.int64)
            names = json.loads(str(data["feature_names"]))
    except (OSError, ValueError) as error:
        if isinstance(error, ConfigValueError):
            raise
        raise ConfigValueError(
            f"cannot read features cache {path}: {error}", value=path
        ) from error
    if version != 1:
        raise ConfigValueError(
            f"features cache {path} has format version {version}, this "
            f"neomd reads version 1", value=path)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ConfigValueError(
            f"features cache {path} holds no usable frames x features matrix "
            f"(shape {values.shape})", value=path)
    if not np.isfinite(values).all():
        raise ConfigValueError(
            f"features cache {path} holds non-finite values (a tape feature "
            f"with unmatched steps writes nan — drop it or fix the tape "
            f"cadence)", value=path)
    return FeatureSet(values=values, steps=steps, run_index=run_index,
                      feature_names=[str(n) for n in names])


# ---------------------------------------------------------------------------
# the model artifact
# ---------------------------------------------------------------------------


def save_model(path, header: Mapping, arrays: Mapping) -> str:
    """Write the versioned model artifact (json header + float64 arrays)."""
    path = os.fspath(path)
    payload = dict(header)
    payload["format_version"] = MODEL_FORMAT_VERSION
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, header=np.array(json.dumps(payload, sort_keys=True)),
             **{key: np.asarray(value, dtype=np.float64)
                for key, value in arrays.items()})
    return path


def load_model(path) -> tuple[dict, dict]:
    """(header dict, {array name: float64 ndarray}) from a model artifact."""
    path = os.fspath(path)
    if not os.path.exists(path):
        raise ConfigValueError(f"model artifact not found: {path}",
                               value=path)
    try:
        loaded = np.load(path, allow_pickle=False)
        if not hasattr(loaded, "files"):  # a bare .npy gives an ndarray
            raise ConfigValueError(
                f"{path} is not a model artifact (a bare array file — "
                f"expected an npz archive with a json 'header')", value=path)
        with loaded as data:
            if "header" not in data.files:
                raise ConfigValueError(
                    f"{path} is not a model artifact (no json header)",
                    value=path)
            header = json.loads(str(data["header"]))
            arrays = {key: np.asarray(data[key], dtype=np.float64)
                      for key in data.files if key != "header"}
    except (OSError, ValueError) as error:
        if isinstance(error, ConfigValueError):
            raise
        raise ConfigValueError(
            f"cannot read model artifact {path}: {error}", value=path
        ) from error
    if not isinstance(header, dict) or \
            header.get("format_version") != MODEL_FORMAT_VERSION:
        found = header.get("format_version") if isinstance(header, dict) \
            else type(header).__name__
        raise ConfigValueError(
            f"model artifact {path} has format version {found!r}, this "
            f"neomd reads version {MODEL_FORMAT_VERSION}", value=path)
    return header, arrays


# ---------------------------------------------------------------------------
# TICA
# ---------------------------------------------------------------------------


def _pooled_covariances(values: np.ndarray, run_index: np.ndarray,
                        mean: np.ndarray, lag: int):
    """(C0, C_tau, n_frames_used, n_pairs) pooled over runs, never crossing
    a run boundary.

    Estimators (documented contract): C0 is the mean-free sample covariance
    over ALL frames with the pooled mean (denominator ``n - n_runs``, the
    one-dof-per-run mean removal); C_tau sums ``outer(x_{t+lag} - mean,
    x_t - mean)`` over every within-run lag pair (denominator ``n_pairs``)
    and is symmetrized as ``(C_tau + C_tau^T)/2`` — the standard TICA
    matrix.
    """
    n, d = values.shape
    runs = np.unique(run_index)
    lengths = {int(run): int((run_index == run).sum()) for run in runs}
    # a run shorter than lag+1 contributes NOTHING to C_tau — pooling would
    # silently ignore it; refuse instead (the user asked for that run)
    starved = [run for run, length in lengths.items() if length <= lag]
    if starved:
        raise ConfigValueError(
            f"lag {lag} leaves no within-run pair in run(s) {starved} "
            f"(run lengths {lengths}); use a smaller --lag",
            key="lag", value=lag)
    c0 = np.zeros((d, d))
    for run in runs:
        block = values[run_index == run] - mean
        c0 += block.T @ block
    c0 /= max(1, n - len(runs))
    c_tau = np.zeros((d, d))
    pairs = 0
    for run in runs:
        block = values[run_index == run]
        head = block[lag:] - mean
        tail = block[:-lag] - mean
        c_tau += head.T @ tail
        pairs += len(head)
    c_tau = 0.5 * (c_tau + c_tau.T) / pairs
    return c0, c_tau, n, pairs


def train_tica(values: np.ndarray, run_index: np.ndarray, feature_names,
               *, lag: int = 1, components: int | None = None,
               ridge: float = 0.0):
    """Fit TICA; returns (header, arrays) for :func:`save_model`.

    TICA semantics: the slow linear combinations solve the generalized
    eigenproblem ``C_tau v = lambda C_0 v`` over the mean-free covariance C0
    and the lag-tau correlation C_tau, via Cholesky whitening (C0 = L L^T,
    ``eigh`` of ``L^-1 C_tau L^-T``, components mapped back through
    ``L^-T``) — no scipy.  Lagged pairs are pooled across runs WITHOUT
    crossing run boundaries (two runs' frames are not one trajectory) and
    the covariance normalization divides by the total pooled pair count.

    ``components`` limits the stored projection to the top-k slowest
    components (all eigenvalues are kept in the header).  ``ridge`` adds
    ``ridge * I`` to C0 when the Cholesky whitening refuses a singular
    covariance.
    """
    values = np.asarray(values, dtype=np.float64)
    n, d = values.shape
    lag = int(lag)
    if lag < 1:
        raise ConfigValueError(f"lag must be >= 1, got {lag}", key="lag",
                               value=lag)
    ridge = float(ridge)
    if ridge < 0.0:
        raise ConfigValueError(f"ridge must be >= 0, got {ridge}",
                               key="ridge", value=ridge)
    k = d if components is None else int(components)
    if not 1 <= k <= d:
        raise ConfigValueError(
            f"components must be in [1, {d}] (the feature count), got {k}",
            key="components", value=k)

    mean = values.mean(axis=0)
    c0, c_tau, n_frames, n_pairs = _pooled_covariances(
        values, run_index, mean, lag)
    c0 = c0 + ridge * np.eye(d)
    try:
        lower = np.linalg.cholesky(c0)
    except np.linalg.LinAlgError as error:
        raise ConfigValueError(
            f"the feature covariance is not positive definite ({error}); "
            f"raise --ridge (e.g. 1e-10) or drop collinear features",
            key="ridge", value=ridge) from error
    # whitened standard eigenproblem: S = L^-1 C_tau L^-T, S u = lambda u
    inv_lower = np.linalg.inv(lower)
    whitened = inv_lower @ c_tau @ inv_lower.T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = np.linalg.eigh(whitened)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    components_matrix = (inv_lower.T @ eigenvectors).T  # rows, whitened back
    header = {
        "model_type": "tica",
        "feature_names": [str(name) for name in feature_names],
        "lag": lag,
        "components": k,
        "ridge": ridge,
        "eigenvalues": [float(v) for v in eigenvalues],
        "n_frames": int(n_frames),
        "n_lag_pairs": int(n_pairs),
        "mean": [float(v) for v in mean],
    }
    arrays = {"mean": mean, "components": components_matrix[:k],
              "eigenvalues": eigenvalues}
    return header, arrays


# ---------------------------------------------------------------------------
# logistic regression
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0.0,
                    1.0 / (1.0 + np.exp(-np.clip(z, -700.0, None))),
                    np.exp(np.clip(z, None, 700.0))
                    / (1.0 + np.exp(np.clip(z, None, 700.0))))


def train_logistic(values: np.ndarray, labels: np.ndarray, feature_names,
                   *, epochs: int = 2000, learning_rate: float = 0.5,
                   l2: float = 0.0):
    """Fit two-basin logistic regression by full-batch gradient descent
    (deterministic: zero init, no shuffling); returns (header, arrays)
    for :func:`save_model`.

    Training runs on standardized features; the mean/std are FOLDED into
    the returned ``weights``/``bias`` so the artifact evaluates as plain
    ``sigmoid(x @ weights + bias)`` in raw feature units.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    n, d = values.shape
    if labels.shape != (n,) or not np.isin(labels, (0, 1)).all():
        raise ConfigValueError(
            "logistic training needs labels in {0, 1} with one per frame "
            f"(got shape {labels.shape}, values "
            f"{sorted(set(np.unique(labels).tolist()))})", key="labels",
            value=labels[:10].tolist())
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    l2 = float(l2)
    if epochs < 1:
        raise ConfigValueError(f"epochs must be >= 1, got {epochs}",
                               key="epochs", value=epochs)
    if learning_rate <= 0.0:
        raise ConfigValueError(
            f"learning_rate must be > 0, got {learning_rate}",
            key="learning_rate", value=learning_rate)
    if l2 < 0.0:
        raise ConfigValueError(f"l2 must be >= 0, got {l2}", key="l2",
                               value=l2)

    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    standardized = (values - mean) / std
    weights = np.zeros(d)
    bias = 0.0
    for _ in range(epochs):
        error = _sigmoid(standardized @ weights + bias) - labels
        grad_w = standardized.T @ error / n + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    # fold the standardization back into raw feature units
    weights_raw = weights / std
    bias_raw = bias - float((mean / std) @ weights)
    probability = _sigmoid(values @ weights_raw + bias_raw)
    prediction = (probability >= 0.5).astype(np.float64)
    accuracy = float((prediction == labels).mean())
    eps = 1e-15
    logloss = float(-(labels * np.log(probability + eps)
                      + (1.0 - labels) * np.log(1.0 - probability + eps)
                      ).mean())
    header = {
        "model_type": "logistic",
        "feature_names": [str(name) for name in feature_names],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "n_frames": int(n),
        "accuracy": accuracy,
        "logloss": logloss,
    }
    arrays = {"weights": weights_raw, "bias": np.array([bias_raw])}
    return header, arrays


# ---------------------------------------------------------------------------
# evaluation + the train entry point
# ---------------------------------------------------------------------------


def apply_model(header: Mapping, arrays: Mapping,
                values: np.ndarray) -> np.ndarray:
    """The one evaluation route for a stored model: TICA projects onto the
    slow components (``(x - mean) @ components.T``), logistic returns
    ``sigmoid(x @ weights + bias)``."""
    model_type = header.get("model_type")
    values = np.asarray(values, dtype=np.float64)
    if model_type == "tica":
        return (values - arrays["mean"]) @ arrays["components"].T
    if model_type == "logistic":
        return _sigmoid(values @ arrays["weights"] + arrays["bias"][0])
    raise ConfigValueError(
        f"unknown model_type {model_type!r}; expected one of "
        f"{list(MODEL_TYPES)}", key="model_type", value=model_type)


def _labels_for_logistic(features: FeatureSet, *, labels_path=None,
                         label_column=None, label_threshold=None) -> np.ndarray:
    """Labels in {0, 1}: from a .npy/.npz array, or by thresholding one
    feature column (the two-basin workflow: label = value > threshold)."""
    if labels_path is not None:
        path = os.fspath(labels_path)
        if not os.path.exists(path):
            raise ConfigValueError(f"labels file not found: {path}",
                                   key="labels", value=path)
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as data:
                if "labels" not in data.files:
                    raise ConfigValueError(
                        f"labels npz {path} has no 'labels' key",
                        key="labels", value=path)
                labels = np.asarray(data["labels"])
        else:
            labels = np.asarray(np.load(path, allow_pickle=False))
        if labels.shape != (features.values.shape[0],):
            raise ConfigValueError(
                f"labels shape {labels.shape} does not match the "
                f"{features.values.shape[0]} frames of the features cache",
                key="labels", value=path)
        return labels.astype(np.float64)
    if label_column is not None:
        if label_threshold is None:
            raise ConfigValueError(
                "--label-column needs --label-threshold", key="label_column",
                value=label_column)
        if label_column not in features.feature_names:
            matches = suggest(label_column, features.feature_names)
            hint = (f"did you mean: {', '.join(matches)}?" if matches else
                    f"features: {features.feature_names}")
            raise ConfigValueError(
                f"label column {label_column!r} is not a feature ({hint})",
                key="label_column", value=label_column)
        column = features.values[:,
                                 features.feature_names.index(label_column)]
        return (column > float(label_threshold)).astype(np.float64)
    raise ConfigValueError(
        "logistic training needs labels: --labels FILE.npy/.npz or "
        "--label-column NAME --label-threshold VALUE", key="labels")


def train(features_path, *, model: str = "tica", output: str = "model.npz",
          lag: int = 1, components: int | None = None, ridge: float = 0.0,
          epochs: int = 2000, learning_rate: float = 0.5, l2: float = 0.0,
          labels_path=None, label_column=None,
          label_threshold=None) -> TrainResult:
    """features.npz -> a trained model artifact (the ``neomd mlcv train``
    library call; hyperparameters map 1:1 from the CLI flags)."""
    if model not in MODEL_TYPES:
        raise _unknown_model_error(model)
    features = load_features(features_path)
    if model == "tica":
        header, arrays = train_tica(
            features.values, features.run_index, features.feature_names,
            lag=lag, components=components, ridge=ridge)
    else:
        labels = _labels_for_logistic(
            features, labels_path=labels_path, label_column=label_column,
            label_threshold=label_threshold)
        header, arrays = train_logistic(
            features.values, labels, features.feature_names,
            epochs=epochs, learning_rate=learning_rate, l2=l2)
    save_model(output, header, arrays)
    diagnostics = {key: header[key] for key in header
                   if key not in ("feature_names", "mean")}
    return TrainResult(output=str(output), model_type=model,
                       feature_names=features.feature_names,
                       n_frames=features.values.shape[0],
                       diagnostics=diagnostics)


def _unknown_model_error(model):
    """The unknown-model-type error (one definition so the CLI message and
    the library message cannot drift apart)."""
    return ConfigKeyError(
        f"unknown model type {model!r}", key="model", value=model,
        known_keys=MODEL_TYPES)
