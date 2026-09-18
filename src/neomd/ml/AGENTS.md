# ML/MM

Preserve mechanical embedding ported from openmm-ml 1.7 and its attribution.
Region forms (indices or residue selectors) are exclusive; cross-boundary
bonded terms remain MM. Assemble before Context creation and never serialize
an NNP Force into System XML.

The model file is the interface: nm input, kJ/mol output. openmm-ml is not a
runtime dependency; the mock path remains usable without torch. The fake kernel
intentionally ignores `ml_region`. For embedding changes consult project-root
`docs/adr/` ADR-0004 and the unit contract in `torchscript.py`.
