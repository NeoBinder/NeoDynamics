# Method execution

Methods prepare biases, resume state and tapes through `PreparedMethod`; the
driver owns stepping and reporting. Keep restraint wiring outside methods.

- SMD reuses restraint `make_bias`; ramp updates substitute parameters at update
  boundaries. Its optional tape trims on resume like other reporting tapes.
- OPES preserves weighted KDE, nearest-kernel compression, explored-region
  normalization and pace-based table pushes. `kernels.npz` is method state,
  written on deposit and replayed with the same math. A reporting probe fires
  too early and would lose the last deposit on resume.
- GaMD calibrates from integrator potentials after zero-strength installation.
  Resume restores saved calibration rather than recalibrating. Keep boost trace
  reporting separate from calibration state; reweighting uses exp(βΔV).
  For algorithm/capability changes consult project-root `docs/adr/` ADR-0005.
