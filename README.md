# NeoDynamics

NeoDynamics is an OpenMM-based Python SDK and command-line tool for preparing,
running and analysing molecular-dynamics simulations. Use YAML plans or Python
calls to run standard MD, enhanced sampling and free-energy calculations, with
checkpoints for continuation and recorded configuration for reproducibility.

Python 3.12+ · OpenMM 8.6.x (default environment) · Linux · [MIT](LICENSE)

<a id="highlights"></a>

- Prepare protein–ligand–solvent systems and inspect their force-field parameters.
- Run minimization, equilibration, MD, metadynamics, steered MD, GaMD/LiGaMD,
  OPES and RBFE windows; analyse trajectories and free-energy estimates.
- Combine MM with TorchScript neural potentials through optional ML/MM support.
- Integrate the SDK into another application through the published contracts.

Method-specific requirements and limits are documented in the
[method guides](docs/methods/index.md). Ligand exchange is experimental;
its trajectories do not establish equilibrium PMFs. See the
[LiGaMD guide](docs/methods/gamd.md) before using it for binding studies.

## Installation

<a id="preferred-pixi--custom-runtime-environment"></a><a id="pixi"></a><a id="pip"></a>

Install the SDK from PyPI (Python 3.12+):

```bash
pip install NeoDynamics
neomd version
```

Component chemistry and protein/system preparation are optional tiers:

```bash
pip install "NeoDynamics[component,preparation]"
```

RBFE windows (`rbfe`) and Torch-backed ML/MM (`mlmm`) are separate extras;
ML/MM also needs the conda-only `openmm-torch` build. For the repository's
locked Linux environment, use [Pixi](https://pixi.sh) instead:

```bash
git clone https://github.com/NeoBinder/NeoDynamics.git
cd NeoDynamics
pixi install
pixi run neomd version
```

See [setup and development](docs/development.md) for environments and extras.

## Quick start

<a id="python--one-facade-three-depths"></a><a id="shell"></a><a id="a-plan-file"></a>

From the repository root, validate and run the bundled 100-step GaMD demo on CPU:

```bash
pixi run neomd validate examples/ligamd2/peptide-smoke.yaml
pixi run neomd run examples/ligamd2/peptide-smoke.yaml --platform cpu
```

This small alanine-dipeptide demo checks the software setup; it is not a binding
simulation. Results go to `/tmp/neomd-ligamd2-peptide-smoke`. The CLI otherwise
defaults to CUDA. The same plan can be run from Python:

```python
from neomd import md_run

md_run("examples/ligamd2/peptide-smoke.yaml", platform="cpu")
```

For your own system, follow the [plan walkthrough](docs/quickstart.md) or the
[complete 3HTB protein–ligand example](examples/3HTB_complex/README.md).

## Documentation

<a id="the-model-in-one-minute"></a><a id="plan--the-immutable-experiment-snapshot"></a>
<a id="the-kernel-seam--kernelport-and-three-adapters"></a><a id="knowledge-triples-and-the-registry"></a>
<a id="driver-probes-sinks--what-a-run-writes"></a><a id="resume--deterministic-continuation"></a>
<a id="system-preparation-and-tools"></a><a id="mlmm-coupling"></a>
<a id="extending-neodynamics"></a><a id="migrating-from-v1"></a>

| I want to… | Start here |
| --- | --- |
| Configure a simulation or migrate a v1 plan | [Quick start](docs/quickstart.md) · [Configuration reference](docs/reference/configuration.md) |
| Prepare a system or inspect parameters | [Preparation and tools](docs/preparation-tools.md) |
| Choose a method and interpret its results | [Methods and analysis](docs/methods/index.md) |
| Find output files or resume a run | [Run outputs and continuation](docs/running.md) |
| Integrate the SDK into an application | [Service and integration surface](docs/integration-surface.md) |
| Understand or extend the SDK | [Architecture](docs/architecture.md) · [Plugins](docs/extending.md) |
| Integrate an algorithm package or Agent | [Algorithm integration](docs/algorithm-integration.md) |

The [documentation home](docs/index.md) links to tutorials, scientific contracts
and scope limits.

## Development

<a id="testing-and-ci"></a>

See [contributor setup and test commands](docs/development.md) and
[contribution constraints](AGENTS.md). Tests cover deterministic fake kernels,
real OpenMM runs and recorded regression samples. See the development guide for
CPU/GPU selection, optional dependency tiers and golden comparison modes.

## License

<a id="versioning-and-license"></a>

Released under the [MIT license](LICENSE). Package versions are derived from Git tags.
