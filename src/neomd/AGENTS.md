# SDK boundaries

- Facade spellings compile to the same immutable, fingerprinted Plan. Compile
  and direct drive share kernel-spec construction. Validation collects all
  errors with key paths and suggestions; validation-only CLI runs write nothing.
- Plugins own declared schema namespaces. Preserve opaque plugin values in
  the fingerprint and discover entry points before Plan construction; an absent
  plugin is an error, not a fallback. See ADR-0002 for namespace changes.
- `resume.py` owns restore and trimming every tape to the checkpoint. Probes
  do not independently choose append/truncate. Coordinate output may wrap
  molecules; checkpoints never wrap because resume must remain bit-exact.
- Keep artifact writes in probes/sinks, OpenMM access behind the kernel seam,
  private API access in `openmm_privates.py`, and torch/openmmtorch imports in
  `ml/` or `mlcv/`. Preserve the source-scan checks for these boundaries.
- `SystemBundle` and geometry QC remain OpenMM-free. QC collects findings before
  a strict-mode failure. External tools use isolated subprocess directories,
  never process-wide `os.chdir`.
- Schema, force expression and observables have one definition per registered
  restraint/CV/method; preserve numpy/kernel parity and registry validation.

For scientific architecture changes, consult the relevant ADR under `docs/adr/`.
For documentation/docstring changes use `docs/AGENTS.md`. These paths are
project-root-relative; local guides carry kernel and method-specific constraints.
