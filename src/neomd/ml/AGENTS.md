# src/neomd/ml — ML/MM (ADR-0004)

`KernelSpec.ml_region` (the barostat-shaped pre-Context assembly spec
`{"indices" | "residues", "model": {"type": "torchscript"|"mock", ...}}` —
the two region forms are mutually exclusive; `residues` selectors
(`CHAIN:RESID` / `CHAIN:NAME`, `selection.py`) resolve against the complex
topology) is assembled by the openmm adapter via `ml.assemble` — mechanical
embedding ported VERBATIM from openmm-ml 1.7 (MIT, attribution in
`embedding.py`) + the NNP force; never written into system.xml (the NNP
Force is not XML-serializable). Cross-boundary bonded terms of residue
regions stay MM (ADR-0004 addendum). openmm-ml is NOT a dependency
(registry rejected; import-gated cross-validation only); the model file is
the interface (nm-in / kJ/mol-out unit contract documented in
`torchscript.py`); the mock NNP keeps the whole pipeline testable without
torch (fake kernel ignores ml_region — documented). torch / openmmtorch
imports live only under `ml/` (source-scanned — see `../AGENTS.md` seam
discipline).
