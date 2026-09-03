# ML/MM ligand demo (3HTB + JZ4 as the ML region)

Wave-2 track W2-d (issue #12 ML/MM part) end-to-end demo: the solvated
3HTB protein with its JZ4 ligand treated by a machine-learning potential —
**mechanical embedding** (ML-ML MM terms removed; the ML atoms keep their MM
charges for the ML↔MM electrostatics; ported verbatim from openmm-ml, see
ADR-0004) — through the public `md_run`/`compile` facade.

Decision pointer: [docs/adr/0004-mlmm-in-tree-coupling.md](../../docs/adr/0004-mlmm-in-tree-coupling.md).
Config keys: `ml_region.indices` (particle indices, ligand-only — W2-d) or
`ml_region.residues` (residue selectors, active-site regions — W3-c; the two
forms are mutually exclusive) + `ml_region.model` (`type: torchscript|mock`).

## Region extent

`--region ligand` (default, W2-d): the JZ4 ligand as a raw index list.
`--region active-site` (W3-c, ADR-0004 addendum): the ligand **plus two
pocket residues** (GLN102, LEU133 — crystal contacts 0.26/0.36 nm) spelled
as residue selectors `["B:JZ4", "A:102", "A:133"]` (grammar: `CHAIN:RESID`
by author numbering or `CHAIN:NAME` by residue name — `neomd.ml.selection`).
Cross-boundary bonded terms — the peptide bonds joining GLN102/LEU133 to
the backbone — **stay in MM** (policy (a), the GROMACS QM/MM convention;
link-atom capping is future work together with true QM/MM). Selectors
resolve against the prepared complex topology; an unmatched selector fails
loudly with a did-you-mean.

## Files

| file | what it is |
|---|---|
| `run_mlmm.py` | the demo: prepare → ml_region → min → MD (uses `md_run`) |
| `build_toy_model.py` | builds the TOY TorchScript NNP (`toy_nnp.pt`) |

## Running it

From the repository root, inside the pinned **ml** pixi environment
(openmm-torch + torch — see ADR-0004 for the environment's temporary
openmm 8.5 pin):

```bash
# full demo: min + 100 ps MD with the toy TorchScript NNP
pixi run -e ml python examples/mlmm_ligand/run_mlmm.py --workdir /tmp/mlmm_demo

# quick taste (2 ps)
pixi run -e ml python examples/mlmm_ligand/run_mlmm.py --workdir /tmp/mlmm_demo --ps 2

# torch-free tier: the mock NNP (runs in any environment, e.g. -e test)
pixi run -e test python examples/mlmm_ligand/run_mlmm.py --workdir /tmp/mlmm_demo --mock --ps 2

# W3-c active-site region: ligand + GLN102/LEU133 as residue selectors
pixi run -e ml python examples/mlmm_ligand/run_mlmm.py --workdir /tmp/mlmm_as --region active-site --ps 2
```

Preparation (~20 s, GAFF via antechamber) is skipped on re-runs once
`sys_prep/3htb/` exists. Outputs land under `--workdir`: `mlmm_min/`,
`mlmm_md/` (state/dcd/checkpoint + `manifest.json` + `last.pdbx`),
`mlmm_report.json`.

## Expected runtime

Measured on the dev box (CPU, `--threads 4`, 31,612 particles, openmm
8.5.2 + openmm-torch 1.5.1 / torch 2.12, dt 1 fs — see the note below):
preparation ~20 s (GAFF; skipped on re-runs), min (200 iters) ~90 s, and
MD at ~113 ms/step with the toy TorchForce (0.78 ns/day) — so the full
default demo (`--ps 100` = 100k steps) is **~3 h**; `--ps 2` (2k steps) is
a ~5-minute taste including the minimization. The mock tier is somewhat
faster (no torch call per step). More threads help roughly linearly.

The MD leg integrates at **dt = 1 fs** (not the repo's usual 2 fs): this
fixture's residual |F|max plateau (~2.4e3 kJ/mol/nm, the known ASN163/LEU164
clash — see examples/3HTB_complex) combined with the toy-NNP tether forces
was observed to NaN at 2 fs and is stable at 1 fs. A real NNP + a properly
equilibrated system should revisit that.

## The toy model is NOT physics

Both model tiers are **pipeline proofs**, not potentials:

* `--mock` — harmonic tethers + soft-sphere repulsion assembled from
  standard openmm custom forces (deterministic, torch-free);
* default — a TorchScript module of per-atom harmonic tethers, run through
  the real openmm-torch `TorchForce`.

**For production, replace the `.pt` file** with a real NNP: the model file
is the interface — no per-model registry, nothing in neomd to change. The
unit contract (documented in `src/neomd/ml/torchscript.py` and
`build_toy_model.py`'s docstring):

* input: the **full system's** positions, `float32` `(N_system, 3)`, in
  **nm** — TorchForce has no atom-subset parameter, so the ML region's
  indices are baked into the model (`index_select` in the toy);
* periodic systems also feed the box vectors `(3, 3)` nm;
* output: scalar energy in **kJ/mol** — Å/eV/kcal-trained models convert
  inside their `forward` (×10 to Å; 1 eV = 96.485 kJ/mol,
  1 kcal/mol = 4.184 kJ/mol).

Periodic systems additionally need
`ml_region.model.long_range_electrostatics` declared (true/false) — whether
the model computes its own PME electrostatics (drives the embedding's
charge handling; see ADR-0004).

## Related tests

* `tests/v2/test_mlmm.py` — mock pipeline (default gate, torch-free),
  TorchScript round-trip + openmm-ml cross-validation (ml env /
  import-gated): `pixi run -e ml test-ml`.
* `tests/v2/test_mlmm_residues.py` — the W3-c residue-selector grammar,
  boundary-bond policy matrix (which bonded terms survive/die, pinned by
  an analytic energy read), torch-tier residue round-trip.
* `tests/v2/test_3htb_e2e.py::test_ml_region_ligand_mock_smoke` — this
  fixture's reduced in-CI smoke.
