# Current scope limits

True QM/MM (ORCA backend, link atoms and charge redistribution) and generic
`min → eq → prod` orchestration are not supported by the current SDK.
Proposals and scheduling belong on the
[issue board](https://github.com/NeoBinder/NeoDynamics/issues).

Supported alternatives have narrower contracts:

- [ML/MM coupling](../methods/mlmm.md) uses in-tree ML-potential regions;
  it is not a QM/MM implementation.
- [RBFE ladders](../methods/rbfe.md) run one full `drive()` per λ window,
  with a runner-level ledger and automatic resume; they are not generic
  multi-leg workflows.
