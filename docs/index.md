# NeoDynamics

**A molecular-dynamics SDK on OpenMM** — generic MD, well-tempered
metadynamics, steered MD, OPES and GaMD behind a single facade, with a
swappable physics kernel.

Python 3.12+ · OpenMM 8.6.x · MIT license · version derived from git tags

The package contains:

- OpenMM pipelines for generic MD (`min` / `eq` / `md` / `prod`) and
  enhanced-sampling methods — well-tempered metadynamics, steered MD,
  OPES, GaMD — plus in-tree ML/MM coupling and RBFE λ-window free-energy
  calculation (QM/MM is planned as a 2.x plugin)
- OpenMM system-building tools (protein + ligand + solvent)
- Post-run analysis (`neomd.analysis`): FES reconstruction, convergence
  windows, block averaging, Tiwary–Parrinello reweighting, BAR/MBAR
- Ligand forcefield creation and support for externally supplied ligand
  forcefields (AM1-BCC/GAFF via antechamber, RESP2 charges via ORCA, or
  expert-designed parameters)

## Highlights

- **One facade, progressive disclosure** — `md_run(dir)` → scalar kwargs →
  full plan dict all compile to an *identical* immutable `Plan` (same sha256
  fingerprint).
- **Validate everything at once** — plan validation collects *all* problems,
  each rendered with its YAML key path and a did-you-mean suggestion.
- **A closed physics seam (`KernelPort`)** with three adapters: `openmm`
  (production), `fake` (deterministic, openmm-free CI), `replay` (golden-tape
  playback) — how the pieces fit together is on the [architecture
  page](architecture.md).
- **Knowledge triples + registry** — restraints, collective variables,
  methods and probes are self-registering triples; plugins register through
  `neomd.register()`.
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
neomd validate plan.yaml --check-files                    # report every problem, write nothing
```

A minimal plan file, the full CLI, a walkthrough of the plan format and
installation options are on the [quick-start page](quickstart.md).

## Documentation

- [Quick start](quickstart.md) — first run, plan-file walkthrough, installation
- [Configuration reference](reference/configuration.md) — every key a plan accepts
- [Tutorials](tutorials/metadynamics.md) — metadynamics, steered MD, restraints
- [Architecture](architecture.md) — how the pieces fit together
- [Development](development.md) — working on NeoDynamics itself

### Methods

One page per method — schema, physics, artifacts and references:

- [GaMD](methods/gamd.md) — Gaussian accelerated MD: BoostOps seam,
  calibration pre-run, dual/LiGaMD modes
- [ML/MM](methods/mlmm.md) — `ml_region` (indices + active-site residue
  selectors) with TorchScript/mock NNP adapters
- [OPES](methods/opes.md) — on-the-fly probability enhanced sampling
  (KDE bias, `kernels.npz` replay)
- [RBFE](methods/rbfe.md) — λ-window ladder: `run_ladder`, du.tsv bands,
  BAR/MBAR analysis
- [Boresch restraint](methods/boresch.md) — orientation anchor over 3+3
  anchor atoms
- [CV library](methods/cv-library.md) — path (s,z) / coordination /
  rmsd-as-CV knowledge triples
- [ML-CV](methods/mlcv.md) — featurize / train / convert CLI (phase 1)
- [Analysis](methods/analysis.md) — enhanced-sampling analysis and
  convergence diagnostics (`neomd.analysis`)
- [QC](methods/qc.md) — structure quality checks (`neomd.qc`)

The full index with issue and ADR cross-references lives at
[methods/index.md](methods/index.md).
