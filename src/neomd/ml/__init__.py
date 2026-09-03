"""neomd.ml — the self-developed ML/MM coupling module (ADR-0004, W2-d).

Decision record (2026-09-02/03): openmm-ml is NOT a dependency.  Only its
generic layer would be useful and only the mechanical-embedding half of that;
the ~65 KB of per-model adapters and the model registry are useless for OUR
TorchScript models (the model file is the interface here).  The unavoidable
base is openmm-torch (TorchForce C++ plugin) + torch, pinned per settled
decision #10.  openmm-ml is demoted to an optional, marker-gated
cross-validation reference (never installed by pixi).

Layout:

* :mod:`neomd.ml.spec`     openmm-free core: raw ``ml_region`` -> frozen
                           :class:`~neomd.ml.spec.MLRegion` (shape, defaults,
                           vocabulary shared with plan validation);
* :mod:`neomd.ml.embedding` mechanical embedding, ported VERBATIM from
                           openmm-ml 1.7 (MIT, attribution in the header);
* :mod:`neomd.ml.mock`     the mock NNP — standard openmm custom forces, a
                           pipeline stand-in (NOT physics) letting the whole
                           pipeline run with no torch installed;
* :mod:`neomd.ml.torchscript` the generic TorchScript loader (TorchForce;
                           nm in / kJ/mol out — units documented there);
* :mod:`neomd.ml.assemble` the adapter-side entry: embedding + NNP force,
                           pre-Context, never in system.xml.

The whole package imports cleanly without openmm AND without torch; the
engine-touching modules (embedding/mock/torchscript/assemble) import openmm
lazily inside their functions, mirroring ``prepare.py``'s boundary convention.
The openmm adapter (``kernel/openmm.py``) is the only caller of
:func:`neomd.ml.assemble.assemble_ml_region`.
"""

from .spec import (
    ML_REGION_KEYS,
    MOCK_DEFAULTS,
    MODEL_KEYS,
    MODEL_TYPES,
    MLRegion,
    parse_ml_region,
)

__all__ = [
    "MLRegion",
    "parse_ml_region",
    "MODEL_TYPES",
    "ML_REGION_KEYS",
    "MODEL_KEYS",
    "MOCK_DEFAULTS",
]
