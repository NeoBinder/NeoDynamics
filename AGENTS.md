# AGENTS.md

Guidance for AI coding agents working in the NeoDynamics repository.

## Project Overview

NeoDynamics (`neomd`) is NeoBinder's open-source Molecular Dynamics (MD) SDK built on
[OpenMM](https://openmm.org/). It provides system-building tools and simulation
pipelines for generic MD, well-tempered metadynamics, QM/MM, and ligand force-field
creation. Licensed under MIT. Authors: Yilang Hu, Xinhao Han.

## Tech Stack

- **Python**: 3.11 only (`requires-python = "==3.11.*"`). Do not introduce 3.12+ syntax.
- **Core engine**: OpenMM 8.2.0, openmmforcefields, OpenFF toolkit / forcefields.
- **Scientific stack** (provided via conda/pixi, not pip): numpy, scipy, rdkit,
  networkx, pandas, mdtraj, MDAnalysis, pdbfixer.
- **Declared pip dependency**: `python-box >= 7.3.2` (used for YAML config as
  attribute-accessible dicts via `Box.from_yaml`).
- **Build**: setuptools + `versioningit` (version derived from git tags). Never
  hard-code `__version__`; it is managed by `versioningit`.
- **External (undeclared) dependency**: `ttk` — required by `bin/convert.py`,
  `bin/hills_ana.py`, and the entire `neomd.qmmm` subpackage. It is not in the env
  specs and must be supplied by the user.

## Environment & Install

Pixi is the preferred environment manager. Conda is also supported.

```bash
# Pixi (preferred)
pixi install
pixi shell

# Conda
conda env create --name neomd -f environment.yaml
conda activate neomd
pip install -e ./
```

**Important — `bin/` script dependencies:** The standalone scripts in `bin/` pull in
extra dependencies (e.g. `ttk`, `pdbfixer`, `MDAnalysis`, `mdtraj`, `pandas`, ORCA/Multiwfn
external binaries for `resp2_orca.py`). **Do NOT install these dependencies on the
user's behalf.** The user installs them themselves. If an agent needs to run a `bin/`
script and a dependency is missing, surface the missing dependency and stop — do not
`pip`/`pixi add` it.

## Running

```bash
python3 bin/prepare_openmm_system.py examples/3HTB_complex/prepare.yaml
python3 bin/run_generic_md.py examples/3HTB_complex/min.yaml
python3 bin/run_generic_md.py examples/3HTB_complex/eq.yaml
python3 bin/run_metadynamics.py examples/ala_meta/meta.yaml
```

Each `bin/` runner takes a YAML `config` as its first positional argument. Generic MD
dispatches on `config.method`: `min`/`minimization`, `eq`/`md`/`equilibration`, or `smd`.
`--platform {cuda,cpu}` (default `cuda`) and `--cuda_index` are supported by the MD runners.

## Repository Layout

```
bin/                Standalone entry-point scripts (see below)
src/neomd/          Importable package (src layout)
  base/             Abstract base classes: BaseEngine, BasePipeline
  builder/          System construction: NeoSystem, ComplexForceField,
                    ligand loading, GAFFTemplateGenerator
  generic/          Standard MD: OpenmmEngine, Pipeline
  metadynamics/     Well-tempered metadynamics: engine, pipeline,
                    collective-variable factories (colvar.py), HILLS writer
  qmmm/             QM/MM (additive/subtractive). Depends on external ttk.
  io/               load_complex / from_openmm / from_amber / from_gromacs, export
  restraints/       Restraint forces + reporters (funnel, distance, angle,
                    dihedral, rmsd, SMD, etc.)
  math/             PBC minimum-image helper
  logger.py, reporters.py, utils.py
examples/           Example YAML configs + tutorials (3HTB_complex, ala_meta)
tests/              pytest-style integration tests running real OpenMM sims
environment.yaml    Conda env definition
pixi.toml           Pixi workspace config (preferred)
pyproject.toml      PEP 621 metadata + setuptools build
```

### `bin/` scripts

| Script | Purpose |
| --- | --- |
| `prepare_openmm_system.py` | Build an OpenMM system (topology, solvation, ions, ligand param) → `system.xml`, `solv.pdbx`, `ligand.json` |
| `run_generic_md.py` | Run minimization / equilibration / SMD via `neomd.generic.Pipeline` |
| `run_metadynamics.py` | Run metadynamics via `MetadynamicsPipeline` |
| `parse_ff_params.py` | Parse bond/angle/torsion params from a serialized OpenMM system XML |
| `ligand_processor.py` | RDKit ligand toolkit: `smiles`, `convert`, `smiles2sdf` subcommands |
| `resp2_orca.py` | RESP2 charge derivation via ORCA + Multiwfn (external binaries) |
| `convert.py` | `.pdbx` → `.pdb` conversion (incl. amber ion renaming) |
| `gethill.py` | Convert a saved numpy colvar array into a PLUMED-style `HILLS` file |
| `hills_ana.py` | Hills/trajectory analysis helpers (library, no CLI) |
| `fix_protein.py` | PDB protein preparation via pdbfixer (library, no CLI) |
| `template_xml_processor.py` | `generate_template` / `modify_template` for ligand template.xml |

## Testing

- Framework: pytest (not declared in env specs — install it in your dev env if needed).
- Single file `tests/test_pipeline.py` with `test_min`, `test_eq`, `test_meta`.
- Tests run real OpenMM simulations on CPU (`platform="cpu"`) and write to
  `tests/data/_test/`. Fixtures live in `tests/data/` (`solv.pdbx`, `system.xml`).
- Run from repo root: `python -m pytest tests/`.

## Code Conventions

There is no automated formatter/linter configured. Match the surrounding code:

- 4-space indentation; mixed single/double quoting is common — follow local file style.
- Import ordering is informal; keep new imports consistent with the file you edit.
- Inline comments in English are preferred; some files contain Chinese comments
  (especially `ligand_processor.py`, `resp2_orca.py`). Follow the file's existing language.
- Configuration flows through YAML → `Box` objects accessed as attributes
  (`config.method`, `config.timestep`, etc.). New options should be read defensively.
- `NeoSystem.from_config(config)` is the central factory for building systems;
  `Pipeline` / `MetadynamicsPipeline` orchestrate engine + reporters + restraints.

## Agent Notes

- This project uses a `src/` layout — the package is `src/neomd`, not `neomd` at repo root.
- `src/neomd/io/convert.py` and `src/neomd/math/__init__.py` are intentionally empty placeholders.
- The QM/MM "additive" engine is spelled `QMMMAddictiveEngine` (file `addictive_qmmm_engine.py`).
  This is a known typo; do not "fix" it without coordinating — it is part of the public API.
- `bin/resp2_orca.py` contains developer-specific hardcoded ORCA paths — leave as-is unless asked.
- When editing YAML examples or documenting restraint types, refer to
  `examples/restraints_in_yaml.md` and the `generate_restraint` dispatcher in
  `src/neomd/restraints/constructor.py` for the authoritative list of supported types.
- Prefer editing existing patterns over introducing new dependencies.
