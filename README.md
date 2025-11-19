# NeoDynamics

NeoDynamics is NeoBinder's open source project for Molecular Dynamics SDK built on top of OpenMM.

NeoDynamics has been tested with generic MD methods and metadynamics.

This package contains:
- OpenMM related pipelines including generic MD, metadynamics (QM/MM, GAMD, and machine learning-powered MD to be released later)
- OpenMM system building tools
- Protein conformation analysis based on OpenMM engine with various forcefields
- Ligand forcefield creation and support for externally supplied ligand forcefields (from AM1BCC, DFT, or expert designed)

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
pixi add "python==3.11.*"
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
# prepare system
python3 /path/to/project/NeoDynamics/bin/prepare_openmm_system.py /path/to/project/NeoDynamics/examples/prep_system.yaml
# generic MD
python3 /path/to/project/NeoDynamics/bin/run_generic_md.py /path/to/project/NeoDynamics/examples/min.yaml
python3 /path/to/project/NeoDynamics/bin/run_generic_md.py /path/to/project/NeoDynamics/examples/eq.yaml
# metadynamics
python3 /path/to/project/NeoDynamics/bin/run_metadynamics.py /path/to/project/NeoDynamics/examples/meta.yaml
```
