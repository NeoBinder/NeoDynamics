# src/neomd/analysis — post-run analysis

Openmm-free post-run analysis of the artifact formats (colvar.tsv /
hills.npz / smd.tsv + the manifest's grid metadata) — WT FES reconstruction
(producer conventions, bit-identical ledger replay), convergence windows,
block averaging, Tiwary–Parrinello reweighting, multi-walker merge —
behind the `neomd analysis` CLI and an importable API other method tracks
consume.
