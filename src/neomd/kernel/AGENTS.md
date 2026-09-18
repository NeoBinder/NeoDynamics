# KernelPort

Optional capabilities are negotiated through `provides()`. The production
adapter owns core OpenMM imports; fake stays deterministic and OpenMM-free;
replay must be registered before factory use.

Allocate force groups through `port.pick_free_force_group`. Restraints and SMD
pulls each share one group per category and report category-total energy;
`independent_force_group` opts an entry into its own group and energy column.
Keep categories separate and the meta/OPES table bias in its own group.

Bias/CV installation respects native system periodicity captured before any
installation. A restraint flag must not make a nonperiodic system periodic.
For boost capability changes consult ADR-0005 under project-root `docs/adr/`.
