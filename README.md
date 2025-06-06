# NeoDynamics

NeoDynamics is NeoBinder's open soruce project for Molecular Dynamics SDK ontop of openmm.

NeoDynamics is tested on method generic md methods and metadynamics.

This Package contains:
- OpenMM related pipelines including generic md, metadynamics. (possibly qmmm/gamd/machine-learning powered md released later.)
- OpenMM system build
- Protein Conformation analysis based on OpenMM engine with various Forcefileds
- Ligand Force Fileds creation and Ligand ForceFileds supplied externally (from AM1BCC / DFT / or expert designed)

# Installation
NeoDynamics can be installed with:
## (*preferred*) By pixi Installation 

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

## By Conda Installation
```bash
mkdir -p /path/to/project
cd /path/to/project
git clone git@github.com:NeoBinder/NeoDynamics.git
cd /path/to/project/NeoDynamics
conda env create --name neomd -f environment.yaml
# devmode installation
conda activate neomd
pip install -e ./
```



## Examples
```bash
# prep_system
python3  /path/to/project/NeoDynamics/bin/prepare_openmm_system /path/to/project/NeoDynamics/examples/prep_system.yaml
# generic md
python3 /path/to/project/NeoDynamics/bin/run_generic_md.py /path/to/project/NeoDynamics/examples/min.yaml
python3 /path/to/project/NeoDynamics/bin/run_generic_md.py /path/to/project/NeoDynamics/examples/eq.yaml
# metadynamics
python3 /path/to/project/NeoDynamics/bin/run_metadynamics.py /path/to/project/NeoDynamics/examples/meta.yaml
```
