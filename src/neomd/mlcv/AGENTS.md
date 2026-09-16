# src/neomd/mlcv — ML-CV phase 1 (ADR-0006)

Numpy-only out-of-tree-style tool (`neomd mlcv featurize|train|convert`) —
features reuse the PUBLIC cv registry's evaluate implementations; TICA
(generalized eigenproblem, runs pooled without crossing boundaries) +
logistic regression, both linear; TorchScript export is torch-gated and
reproduces `apply_model` bit-tightly. ZERO simulation-core changes —
phase 2 (TorchCV injection through the kind-driven CVIR precedent) is
designed in ADR-0006.
