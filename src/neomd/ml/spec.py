"""spec — the openmm-free core of the ML/MM coupling module (ADR-0004).

``ml_region`` arrives as a raw mapping (the plan section, verbatim, carried by
:class:`~neomd.kernel.port.KernelSpec` exactly like ``barostat``).  This module
normalizes it into an immutable :class:`MLRegion` with NO engine imports, so
plan validation and any kernel can inspect the region without openmm/torch.

The plan-level validation (collect-all, yaml key path + did-you-mean) lives in
:mod:`neomd.plan`; :func:`parse_ml_region` re-checks the same shape at the
assembly boundary as a defensive second gate (a hand-built ``KernelSpec``
bypasses the Plan validator — the adapter must not trust its input).

Shape (ADR-0004)::

    ml_region:
      indices: [i, j, ...]        # 0-based particle indices, ligand-only (W2-d)
      model:
        type: torchscript | mock
        path: model.pt            # torchscript only; the model file IS the
                                  # interface (no per-model registry)
        long_range_electrostatics: false   # torchscript only (see embedding)
        periodic: true            # optional; defaults to the system's
        tether_k: 500.0           # mock only (kJ/mol/nm^2)
        repulsion_k: 1.0          # mock only (kJ/mol)
        repulsion_sigma: 0.15     # mock only (nm)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import ConfigKeyError, ConfigValueError

__all__ = ["MLRegion", "MODEL_TYPES", "ML_REGION_KEYS", "MODEL_KEYS",
           "flatten_indices", "parse_ml_region"]

#: accepted model.type values (the vocabulary plan validation reports)
MODEL_TYPES = ("mock", "torchscript")

#: accepted ml_region keys
ML_REGION_KEYS = frozenset({"indices", "model"})

#: accepted model keys (union over model types; per-type required sets below)
MODEL_KEYS = frozenset({
    "type",
    "path",                     # torchscript
    "long_range_electrostatics",  # torchscript
    "periodic",                 # both (optional)
    "tether_k", "repulsion_k", "repulsion_sigma",  # mock
})

#: keys each model type requires beyond "type"
REQUIRED_MODEL_KEYS = {
    "torchscript": ("path",),
    "mock": (),
}

#: mock NNP defaults (documented as a pipeline mock, NOT physics)
MOCK_DEFAULTS = {
    "tether_k": 500.0,        # kJ/mol/nm^2
    "repulsion_k": 1.0,       # kJ/mol
    "repulsion_sigma": 0.15,  # nm
}


def flatten_indices(value) -> list[int]:
    """index-list spelling (int | numeric str | comma-joined str | list) -> ints.

    Mirrors the restraint ``grp`` conventions (the 3HTB fixture spells index
    groups as comma strings), so ``"12,13,14"`` and ``[12, 13, 14]`` mean the
    same region.  Non-index entries yield -1 so the caller reports them.
    """
    if isinstance(value, bool):
        return [-1]
    if isinstance(value, int):
        return [value]
    if isinstance(value, float):
        return [int(value)] if float(value).is_integer() else [-1]
    if isinstance(value, str):
        out = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                out.append(int(item))
            except ValueError:
                return [-1]
        return out or [-1]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(flatten_indices(item))
        return out
    return [-1]


@dataclass(frozen=True)
class MLRegion:
    """Normalized ML region (openmm-free; consumed by the adapter assembly).

    ``indices``: sorted unique 0-based particle indices of the ML region.
    ``model_type``: ``"mock"`` or ``"torchscript"``.
    ``params``: the model section verbatim (path, mock knobs, ...).
    """

    indices: tuple[int, ...]
    model_type: str
    params: dict

    @property
    def long_range_electrostatics(self) -> bool | None:
        """The mlLongRange declaration of the source port.

        The mock computes no electrostatics at all — declared ``False`` by
        construction.  For torchscript the plan's declaration is passed
        through: ``None`` = undeclared, and for a periodic system the
        embedding refuses (verbatim source behavior — an ML model's
        long-range use cannot be probed); for a non-periodic system it is
        moot.
        """
        if self.model_type == "mock":
            return False
        declared = self.params.get("long_range_electrostatics")
        return None if declared is None else bool(declared)


def _problem(path: tuple, message: str, key=None, known_keys=None):
    """Build the error the way plan.py's validator does (path + did-you-mean)."""
    exc = ConfigKeyError if known_keys is not None else ConfigValueError
    return exc(message, key=key if key is not None else path[-1],
               known_keys=known_keys)


def parse_ml_region(raw) -> MLRegion:
    """Raw ml_region mapping -> :class:`MLRegion` (raises on any bad shape).

    The FIRST problem raises (this is the defensive boundary gate, not the
    collect-all pass — collect-all belongs to Plan validation, which normal
    runs always pass through first).
    """
    if not isinstance(raw, Mapping):
        raise _problem(("ml_region",),
                       f"ml_region must be a mapping with 'indices' and "
                       f"'model', got {type(raw).__name__}")
    for key in raw:
        if key not in ML_REGION_KEYS:
            raise _problem(("ml_region", key),
                           f"unknown ml_region key {key!r}",
                           key=key, known_keys=ML_REGION_KEYS)
    if "indices" not in raw or "model" not in raw:
        missing = "indices" if "indices" not in raw else "model"
        raise _problem(("ml_region", missing),
                       f"ml_region requires {missing!r}",
                       key=missing, known_keys=ML_REGION_KEYS)

    indices = flatten_indices(raw["indices"])
    if not indices or indices[0] < 0:
        raise _problem(
            ("ml_region", "indices"),
            "ml_region.indices must be a non-empty list of 0-based particle "
            "indices (ints, or the comma-string spelling)")
    if any(i < 0 for i in indices):
        raise _problem(("ml_region", "indices"),
                       "ml_region.indices must be non-negative")
    indices = tuple(sorted(set(indices)))

    model = raw["model"]
    if not isinstance(model, Mapping) or not isinstance(model.get("type"), str):
        raise _problem(
            ("ml_region", "model"),
            "ml_region.model must be a mapping with a string 'type' "
            f"(one of {list(MODEL_TYPES)})")
    model_type = model["type"]
    if model_type not in MODEL_TYPES:
        raise _problem(("ml_region", "model", "type"),
                       f"unknown ml_region model type {model_type!r}",
                       key="type",
                       known_keys=frozenset(MODEL_TYPES))
    for key in model:
        if key not in MODEL_KEYS:
            raise _problem(("ml_region", "model", key),
                           f"unknown ml_region.model key {key!r}",
                           key=key, known_keys=MODEL_KEYS)
    for required in REQUIRED_MODEL_KEYS[model_type]:
        value = model.get(required)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise _problem(
                ("ml_region", "model", required),
                f"ml_region.model type {model_type!r} requires {required!r}",
                key=required, known_keys=MODEL_KEYS)

    params = dict(model)
    if "periodic" in params and not isinstance(params["periodic"], bool):
        raise _problem(("ml_region", "model", "periodic"),
                       "ml_region.model.periodic must be a boolean", key="periodic")
    if ("long_range_electrostatics" in params
            and not isinstance(params["long_range_electrostatics"], bool)):
        raise _problem(
            ("ml_region", "model", "long_range_electrostatics"),
            "ml_region.model.long_range_electrostatics must be a boolean",
            key="long_range_electrostatics")
    return MLRegion(indices=indices, model_type=model_type, params=params)
