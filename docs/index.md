# NeoDynamics

**A molecular-dynamics SDK on OpenMM** — generic MD, well-tempered
metadynamics and steered MD behind a single facade, with a swappable
physics kernel.

Python 3.12+ · OpenMM 8.6.x · MIT license · version derived from git tags

The package contains:

- OpenMM pipelines for generic MD (`min` / `eq` / `md` / `prod`),
  well-tempered metadynamics and steered MD (QM/MM, GaMD and ML-powered MD
  are planned as 2.x plugins)
- OpenMM system-building tools (protein + ligand + solvent)
- Ligand forcefield creation and support for externally supplied ligand
  forcefields (AM1-BCC/GAFF via antechamber, RESP2 charges via ORCA, or
  expert-designed parameters)

## Highlights

- **One facade, progressive disclosure** — `md_run(dir)` → scalar kwargs →
  full plan dict all compile to an *identical* immutable `Plan` (same sha256
  fingerprint).
- **Validate everything at once** — plan validation collects *all* problems
  (not fail-on-first), each rendered with its YAML key path and a
  did-you-mean suggestion.
- **A closed physics seam (`KernelPort`)** with three adapters: `openmm`
  (production), `fake` (deterministic, openmm-free, millisecond CI),
  `replay` (golden-tape playback pinning bit-exact parity with v1).
- **Knowledge triples + registry** — restraints, collective variables,
  methods and probes are self-registering triples (schema + physics +
  observables); plugins register through `neomd.register()`.
- **Deterministic resume** — a single owner restores the checkpoint and
  trims every tape to the resume step.

## Quick start

```python
from neomd import md_run

md_run("path/to/run_dir")                # L0: zero-config (reads the plan file in the dir)
md_run("path/to/run_dir", steps=50000)   # L1: scalar knobs deepen the plan
md_run(plan_dict)                        # L2: the full experiment spec
```

```bash
neomd run path/to/run_dir --steps 50000 --platform cpu   # --kernel openmm|fake|replay
neomd prepare prep_config.yaml                            # system preparation (protein+ligand+solvent)
neomd migrate old_v1_config.yaml -o plan.yaml             # one-shot v1 YAML -> Plan translation
neomd validate plan.yaml --check-files                    # report every problem, write nothing
```

A minimal plan file:

```yaml
method: eq
steps: 5000

integrator:
  dt: 0.002
  friction_coeff: 1.0
barostat:
  frequency: 25
  pressure: 1.0
temperature: 298
seed: 0

input_files:
  complex: /work_dir/min/last.pdbx        # from the previous leg
  system: /work_dir/sys_prep/htb/system.xml
  ligands: /work_dir/sys_prep/htb/ligand.json

output:
  output_dir: /work_dir/eq
  trajectory_interval: 1000
  checkpoint_interval: 1000
```

Every key a plan accepts is listed in the [configuration
reference](reference/configuration.md). Metadynamics and steered MD are
covered by the [tutorials](tutorials/metadynamics.md); how the pieces fit
together is on the [architecture page](architecture.md).

## Installation

Pixi (preferred):

```bash
git clone git@github.com:NeoBinder/NeoDynamics.git
cd NeoDynamics
pixi install
pixi shell
```

Or install into your own environment from git:

```bash
pixi add --pypi "neodynamics @ git+https://github.com/NeoBinder/NeoDynamics"
```

See the [README](https://github.com/NeoBinder/NeoDynamics#installation) for
all options (pixi, conda + editable install) and the [development
page](development.md) for working on NeoDynamics itself.
