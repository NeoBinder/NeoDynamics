# NeoDynamics

NeoDynamics is NeoBinder's open source project for Molecular Dynamics SDK built on top of OpenMM.

NeoDynamics has been tested with generic MD methods and metadynamics.

This package contains:
- OpenMM related pipelines including generic MD, metadynamics (QM/MM, GAMD, and machine learning-powered MD to be released later)
- OpenMM system building tools
- Protein conformation analysis based on OpenMM engine with various forcefields
- Ligand forcefield creation and support for externally supplied ligand forcefields (from AM1BCC, DFT, or expert designed)

## Quick start (v2: `md_run`)

Since v0.2.0 the v2 architecture is the default: an immutable `Plan`, a
kernel seam (`openmm` / `fake` / `replay` adapters), knowledge triples
registered via `register()`, and the single facade `md_run` with progressive
disclosure:

```python
from neomd import md_run

md_run("path/to/run_dir")                      # L0: zero-config (reads the plan file in the dir)
md_run("path/to/run_dir", steps=50000)         # L1: scalar knobs deepen the plan
md_run(plan_dict)                              # L2: the full experiment spec
```

The same spellings work from the shell:

```bash
neomd run path/to/run_dir --steps 50000        # the [project.scripts] entry point
neomd prepare prep_config.yaml                 # system preparation (protein+ligand+solvent)
neomd migrate old_v1_config.yaml -o plan.yaml  # one-shot v1 YAML -> Plan translation
```

# Installation
NeoDynamics can be installed using:
## (*preferred*) Pixi Installation

1. Install [pixi](https://pixi.sh/latest/#alternative-installation-methods)
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### Custom Runtime Environment
2. Install package
```bash
mkdir -p /path/to/env
cd /path/to/env
pixi init neomd
cd neomd
pixi add "python==3.12.*"
# git installation
pixi add --pypi "neodynamics @ git+https://github.com/NeoBinder/NeoDynamics"
# local installation
pixi add --pypi "neodynamics @ file:///path/to/NeoDynamics"

pixi add my_custom_conda_package
pixi add --pypi my_custom_pypi_package
pixi shell
```

### Development Environment
2. Install package
```bash
git clone git@github.com:NeoBinder/NeoDynamics.git
cd NeoDynamics
pixi install
pixi shell
```

## Conda Installation
```bash
mkdir -p /path/to/project
cd /path/to/project
git clone git@github.com:NeoBinder/NeoDynamics.git
cd /path/to/project/NeoDynamics
conda env create --name neomd -f environment.yaml
# development mode installation
conda activate neomd
pip install -e ./
```



## Examples
```bash
# prepare system (protein + ligand + solvent, GAFF via antechamber)
neomd prepare /path/to/project/NeoDynamics/examples/3HTB_complex/prepare.yaml
# generic MD (a prepared system + a plan file)
neomd run /path/to/work_dir/min --platform cpu
neomd run /path/to/work_dir/eq --platform cpu
# metadynamics: method: metadynamics in the plan; same facade
neomd run /path/to/work_dir/meta --platform cpu
```

The runnable v2 walkthrough for the 3HTB complex lives in
[examples/3HTB_complex/run_v2.py](examples/3HTB_complex/run_v2.py)
(prepare -> translate -> `md_run`, with smoke presets).

The old v1 bin/ entry points (`run_generic_md.py`, `run_metadynamics.py`,
`prepare_openmm_system.py`, ...) remain as thin compatibility wrappers over
the new CLI for one release; `src/neomd_legacy/` holds the frozen v1 package
during the deprecation window.
