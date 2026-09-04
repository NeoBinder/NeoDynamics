# Quick start

## Python — one facade, three depths

```python
from neomd import md_run

md_run("path/to/run_dir")                # L0: zero-config (reads the plan file in the dir)
md_run("path/to/run_dir", steps=50000)   # L1: scalar knobs deepen the plan
md_run(plan_dict)                        # L2: the full experiment spec
```

The **round-trip law**: L0, L1 and L2 spellings of the same experiment
compile to an identical `Plan` — one validation path, one sha256
fingerprint.

## Shell

```bash
neomd run path/to/run_dir --steps 50000 --platform cpu   # --kernel openmm|fake|replay
neomd prepare prep_config.yaml                            # system preparation (protein+ligand+solvent)
neomd migrate old_v1_config.yaml -o plan.yaml             # one-shot v1 YAML -> Plan translation
neomd validate plan.yaml --check-files                    # report every problem, write nothing
```

## A plan file

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
reference](reference/configuration.md).

## Beyond plain MD

Metadynamics swaps in `method: metadynamics`, a `colvars:` section
(1–3 collective variables from the CV vocabulary) and a `meta_set:`
section (`biasFactor`, `height`, `frequency`) — same facade, same
artifacts plus `colvar.tsv`, `hills.npz` and `fes.tsv`. See the
[metadynamics tutorial](tutorials/metadynamics.md).

Steered MD swaps in `method: smd` and an `smd:` section whose entries
use the restraint vocabulary — any rampable key (`restr_k`, `max_nm`,
`min_degree`, `order`, `maxRMSD_nm`, `ref_position_nm`, ...) given a
LIST of values is piecewise-linearly interpolated over `steps` and
pushed to the kernel on a fixed 5000-step staircase (v1 semantics,
verbatim). A classic pull is a `max_nm`/`ref_position_nm` ramp; a soft
engage/release is a `restr_k` ramp like `[0, 1000, ..., 0]`. See the
[steered-MD tutorial](tutorials/steered-md.md).

Complete, runnable examples:
[examples/3HTB_complex/](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/3HTB_complex)
(protein–ligand complex) and
[examples/ala_meta/](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/ala_meta)
(alanine-dipeptide metadynamics).

## Installation

### (*preferred*) Pixi — custom runtime environment

1. Install [pixi](https://pixi.sh/latest/#alternative-installation-methods)
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

2. Install the package into your own environment
```bash
mkdir -p /path/to/env && cd /path/to/env
pixi init neomd && cd neomd
pixi add "python==3.12.*"
# git installation
pixi add --pypi "neodynamics @ git+https://github.com/NeoBinder/NeoDynamics"
# local installation
pixi add --pypi "neodynamics @ file:///path/to/NeoDynamics"

pixi add my_custom_conda_package
pixi add --pypi my_custom_pypi_package

pixi shell
```

### Pixi — development environment

```bash
git clone git@github.com:NeoBinder/NeoDynamics.git
cd NeoDynamics
pixi install
pixi shell
```

### Conda

```bash
git clone git@github.com:NeoBinder/NeoDynamics.git
cd NeoDynamics
conda env create --name neomd -f environment.yaml
conda activate neomd
pip install -e ./    # development-mode installation
```

For working on NeoDynamics itself, see the [development
page](development.md).
