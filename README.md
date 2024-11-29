# NeoDynamics

NeoDynamics is NeoBinder's open soruce project for Molecular Dynamics SDK ontop of openmm.

NeoDynamics is tested on method generic md methods and metadynamics.

This Package contains:
- OpenMM related pipelines including generic md, metadynamics. (possibly qmmm/gamd relased later.)
- OpenMM system build
- Protein Conformation analysis based on OpenMM engine with various Forcefileds
- Ligand Force Fileds creation and Ligand ForceFileds supplied externally (from AM1BCC / DFT / or expert designed)

## Installation
NeoDynamics can be installed with:

* source code installation
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

* pypi
to be published

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