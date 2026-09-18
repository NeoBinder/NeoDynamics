# Scientific verification

Prefer public-interface tests; use a focused internal test when it uniquely
checks a meaningful invariant. Select relevant Pixi test tasks from
`pyproject.toml`, including golden, legacy or ML tiers when affected.

- Golden bit-exact comparisons require the recording microarchitecture.
  Cross-machine CI uses `NEO_GOLDEN_TOLERANT=1`: max 1e-3 / mean 1e-4 kJ/mol,
  statistics rtol 1e-3, without coordinate-hash identity. Apply the selected
  comparison mode at every step, including step zero; require nonempty samples
  and equal energy/frame sample counts.
- ML/MM analytical energies use the pinned OpenMM Coulomb constant and retain
  CPU rtol=1e-6 and Reference rtol=1e-12 checks.
- Mark actual CUDA execution, including implicit default-platform use, so
  CPU-only runners can skip it without disabling explicit CPU or fake tests.
- Keep pre-commit checks nonmutating for the frozen legacy tree and example
  data; lint exclusions remain in project configuration.
