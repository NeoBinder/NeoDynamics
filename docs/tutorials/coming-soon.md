# Coming soon

!!! warning "Not yet implemented"

    The features below are **planned but not shipped**. They are tracked
    on the [issue board](https://github.com/NeoBinder/NeoDynamics/issues).
    Everything on the other tutorial and method pages works today.

## QM/MM

True QM/MM (ORCA backend, link atoms, charge redistribution), to be
rebuilt as a 2.x plugin with two real adapters (production QM backend +
mock). The ML/MM coupling (`ml_region`, see [methods/mlmm.md](../methods/mlmm.md))
already covers ML-potential regions in-tree.

## Multi-leg orchestration

`min → eq → prod` in one invocation is deliberately deferred to 2.x.
The RBFE ladder (`neomd.rbfe.run_ladder`, see [methods/rbfe.md](../methods/rbfe.md))
is today's deliberately narrow special case: one full `drive()` per λ
window, with a runner-level ledger and automatic resume.
