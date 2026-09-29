# NeoDynamics documentation

NeoDynamics prepares, runs and analyses OpenMM molecular-dynamics simulations
through Python and the `neomd` command line. Start with a working example, then
choose the preparation workflow and scientific method for your system.

## Quick start

For installation and a bundled CPU demo, start with the
[project README](https://github.com/NeoBinder/NeoDynamics#quick-start).
The [quick-start guide](quickstart.md) explains YAML plans, Python calls,
validation and migration from v1.

## User guides

<a id="highlights"></a><a id="documentation"></a>

| Task | Guide |
| --- | --- |
| Prepare protein, ligand and solvent inputs | [Preparation and tools](preparation-tools.md) |
| Inspect every accepted plan option | [Configuration reference](reference/configuration.md) |
| Read results, control wrapping or continue a run | [Outputs and continuation](running.md) |
| Learn a sampling setup | [Metadynamics](tutorials/metadynamics.md) · [Steered MD](tutorials/steered-md.md) · [Restraints](tutorials/restraints.md) |

Preparation details branch into [component chemistry review](component-inspection.md),
[protein identity and repair](protein-preparation.md) and
[verified prepared bundles](prepared-bundle.md). These pages distinguish
supported SDK workflows from service intake and remote operation limits.
Ligand package methods include [generated GAFF](component-gaff.md) and
[OpenFF Sage 2.2.1](component-openff.md), each with separate resource and
assembly qualification.
The [SEP peptide profile](component-sep.md) retains a modified residue inside
one reviewed protein chain and qualifies its separate full-ff14SB template family.

## Methods

For algorithm packages, workflows and Agent boundaries, see
[algorithm integration](algorithm-integration.md).

The [method index](methods/index.md) links to GaMD/LiGaMD, OPES, RBFE,
ML/MM, collective variables, analysis and quality checks. Each method page
covers configuration, outputs, continuation and scientific references.

For GaMD, read both the [method guide](methods/gamd.md) and
[CE2 analysis requirements](methods/gamd-analysis.md). Deterministic ligand
exchange is experimental and does not establish an equilibrium PMF.
See [scope limits](tutorials/coming-soon.md) for unsupported capabilities.

## Interactive service

Start with the [resident service guide](service-runtime.md) for configuration,
process isolation, admission, checkpoints and configured publication. Its
linked contracts cover [scientific protocols](scientific-protocol.md),
[durable command outcomes](service-ledger.md),
[HTTP/WebSocket transport](private-transport.md) and
[trajectory result evidence](trajectory-results.md) and
[bounded comparisons](trajectory-comparisons.md).

For embedded control, see the [interactive driver](interactive-driver.md).
Check [runtime qualification](runtime-qualification.md) for what has actually
been tested, including scale limits and outstanding gates.

## Development

- [Setup and checks](development.md): environments, tests, golden modes and docs builds.
- [Architecture](architecture.md): plans, kernels, reporting and ownership boundaries.
- [Extensions](extending.md): method plugins, entry points and custom reporting.
- [Architecture decisions](adr/0001-neomd2-strangler-migration.md): rationale and retained scope decisions.
