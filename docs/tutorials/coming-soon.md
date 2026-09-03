# Coming soon

!!! warning "Not yet implemented"

    The features below are **planned but not shipped**. They are tracked
    on the [issue board](https://github.com/NeoBinder/NeoDynamics/issues)
    and the development plan (`docs/issue-dev-plan.md` in the repository).
    Everything on the other tutorial pages works today.

## OPES

On-the-fly probability enhanced sampling (#11) as a registry method
(`methods/opes.py`) mirroring the metadynamics seam: KDE → bias table →
`update_table`, with a `kernels.npz` artifact and resume replay.
Standard and explore modes are planned.

## GaMD

Gaussian-accelerated MD (#10) as a real plugin on the verified plugin
seam (the `examples/gamd_drill/` drill already proves registration,
discovery and dispatch): boost calibration through `energy_forces()`,
LiGaMD group boosts via `GroupEnergy`, online parameter updates via
`BiasParamOps`.

## RBFE

Relative binding free energy (#8): a Boresch restraint triple, λ-window
orchestration (the first real customer of multi-leg orchestration,
deferred to 2.x), softcore perturbation at the kernel seam, and BAR/MBAR
analysis.

## Analysis toolkit

A `neomd.analysis` subpackage + `neomd analysis` CLI (#16) for the v2
artifact formats (`colvar.tsv`, `hills.npz`, `smd.tsv`): FES convergence,
block averaging, Tiwary–Parrinello reweighting, multi-walker merging.

## Also planned

QM/MM and ML-powered MD are planned as 2.x plugins (see the README's
extension section); multi-leg orchestration (`min → eq → prod` in one
invocation) is deliberately deferred to 2.x.
