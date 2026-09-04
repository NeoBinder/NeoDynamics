"""spec — the openmm-free core of the ML/MM coupling module (ADR-0004).

Raw ``ml_region`` mapping -> frozen MLRegion: ``indices`` and ``residues``
are MUTUALLY EXCLUSIVE forms; model keys/defaults in parse_ml_region, which
re-checks the plan-validated shape as a defensive second gate.  Reference:
docs/methods/mlmm.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..errors import ConfigKeyError, ConfigValueError
from .selection import is_residue_selector, resolve_residues

__all__ = ["MLRegion", "MODEL_TYPES", "ML_REGION_KEYS", "MODEL_KEYS",
           "flatten_indices", "flatten_selectors", "parse_ml_region"]

#: accepted model.type values (the vocabulary plan validation reports)
MODEL_TYPES = ("mock", "torchscript")

#: accepted ml_region keys
ML_REGION_KEYS = frozenset({"indices", "residues", "model"})

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


def flatten_selectors(value) -> list[str]:
    """residues spelling (selector str | comma-joined str | list) -> selectors.

    Mirrors :func:`flatten_indices`' tolerance (``"A:JZ4,A:29"`` and
    ``["A:JZ4", "A:29"]`` mean the same region).  Non-selector entries come
    back verbatim so the caller's grammar check can report them.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(flatten_selectors(item))
        return out
    return [value]  # scalar garbage — reported by the grammar check verbatim


@dataclass(frozen=True)
class MLRegion:
    """Normalized ML region (openmm-free; consumed by the adapter assembly).

    ``indices``: sorted unique 0-based particle indices of the ML region —
    verbatim from the ``indices`` spelling, or RESOLVED from the ``residues``
    selectors against the system topology (W3-c: active-site regions).
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


def _problem(path: tuple, message: str, key=None, known_keys=None, value=None):
    """Build the error the way plan.py's validator does (path + did-you-mean)."""
    exc = ConfigKeyError if known_keys is not None else ConfigValueError
    kwargs = {"key": key if key is not None else path[-1],
              "known_keys": known_keys}
    if value is not None:
        kwargs["value"] = value
    return exc(message, **kwargs)


def parse_ml_region(raw, topology=None) -> MLRegion:
    """Raw ml_region mapping -> :class:`MLRegion` (raises on any bad shape).

    The FIRST problem raises (this is the defensive boundary gate, not the
    collect-all pass — collect-all belongs to Plan validation, which normal
    runs always pass through first).

    ``topology``: required to RESOLVE the ``residues`` form (anything with
    the ``openmm.app.Topology`` atom surface — the adapter passes the loaded
    complex structure's topology); the ``indices`` form ignores it.
    """
    if not isinstance(raw, Mapping):
        raise _problem(("ml_region",),
                       f"ml_region must be a mapping with 'indices' or "
                       f"'residues' and 'model', got {type(raw).__name__}")
    for key in raw:
        if key not in ML_REGION_KEYS:
            raise _problem(("ml_region", key),
                           f"unknown ml_region key {key!r}",
                           key=key, known_keys=ML_REGION_KEYS)
    has_indices, has_residues = "indices" in raw, "residues" in raw
    has_model = "model" in raw
    if not (has_indices or has_residues) or not has_model:
        # 'indices' or 'residues' (either form) AND 'model' — rendered in
        # that grammar, with a missing piece as the key-path leaf
        region_part = "'indices' or 'residues' (the ML region's atoms)"
        if not (has_indices or has_residues) and not has_model:
            message, leaf = f"ml_region requires {region_part}, and 'model'", "indices"
        elif not has_model:
            message, leaf = "ml_region requires 'model'", "model"
        else:
            message, leaf = f"ml_region requires {region_part}", "indices"
        raise _problem(("ml_region", leaf), message,
                       key=leaf, known_keys=ML_REGION_KEYS)
    if has_indices and has_residues:
        raise _problem(
            ("ml_region",),
            "ml_region takes EITHER 'indices' OR 'residues', not both — a "
            "region defined two ways invites a stale list silently surviving "
            "a switch of spelling (ADR-0004 W3-c addendum)",
            known_keys=ML_REGION_KEYS)

    if has_indices:
        indices = flatten_indices(raw["indices"])
        if not indices or indices[0] < 0:
            raise _problem(
                ("ml_region", "indices"),
                "ml_region.indices must be a non-empty list of 0-based particle "
                "indices (ints, or the comma-string spelling)")
        if any(i < 0 for i in indices):
            raise _problem(("ml_region", "indices"),
                           "ml_region.indices must be non-negative")
        resolved = tuple(sorted(set(indices)))
    else:
        selectors = flatten_selectors(raw["residues"])
        bad = [s for s in selectors if not is_residue_selector(s)]
        if not selectors or bad:
            raise _problem(
                ("ml_region", "residues"),
                "ml_region.residues must be a non-empty list of 'CHAIN:RESID' "
                "/ 'CHAIN:NAME' selectors (e.g. \"A:29\", \"A:JZ4\"; the "
                "comma-string spelling is accepted)", value=bad[0] if bad else None)
        if topology is None:
            raise _problem(
                ("ml_region", "residues"),
                "ml_region.residues must be resolved against the system "
                "topology — this entry point received none (resolution "
                "happens at neomd validate --check-files and at the openmm "
                "adapter's assembly)")
        resolved = tuple(resolve_residues(selectors, topology))

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
    return MLRegion(indices=resolved, model_type=model_type, params=params)
