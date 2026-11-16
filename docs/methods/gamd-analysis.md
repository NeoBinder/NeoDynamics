# GaMD CE2 analysis

`neomd analysis gamd-ce2` reads GaMD saved-frame artifacts directly; it does not
reconstruct metadynamics hills. Analysis uses NumPy and the standard library,
without importing OpenMM. See [GaMD sampling](gamd.md) for run configuration.

For each coordinate bin, it first sums all recorded boost increments
`D = sum(channel__boost)`, then computes population moments (`ddof=0`):

```text
beta = 1 / (R * temperature)
log_weight = beta * mean(D) + beta² * variance(D) / 2
F = F_biased - mean(D) - beta * variance(D) / 2 + constant
```

`R = 0.00831446261815324 kJ/(mol K)`. Summing before computing variance retains
cross-channel covariance. Original `channel__energy` values are not boost
increments. CE2 is a conditional ensemble calculation, not a per-frame
second-order weight. The target ensemble retains user restraints.

## Run the analysis

Save this as `analysis.json`, replacing the groups and bin edges for your system:

```json
{
  "temperature": 298,
  "coordinates": [
    {"kind": "distance", "groups": [[0, 1, 2], [10, 11, 12]], "pbc": "minimum_image"}
  ],
  "edges": [[0, 0.2, 0.4, 0.6, 0.8, 1.0]],
  "min_count": 10,
  "n_blocks": 4
}
```

```bash
neomd analysis gamd-ce2 run --config analysis.json --out pmf.npz
```

By default the command reads `run/production.dcd`, `gamd.tsv` and
`gamd_metadata.json`. A staged run keeps one DCD per stage (ADR-0010), and
production's is the CE2 frame source.  Since ADR-0011 every stage (including
`production`) is a leg whose step axis and DCD first step start at 0, so
cross-stage frame indices are never comparable — read production frames from
`production.dcd` alone. The SDK DCD header supplies actual first
step and stride;
its timestep must match full-precision metadata within float32 precision.
Analysis rejects incomplete frames, trailing bytes, missing matching boost rows,
invalid box records and incompatible artifact schemas. It selects only
`production` rows with one frozen parameter revision. Extra GaMD report rows
between trajectory frames are joined by step, never paired by row number.

Explicit-selection runs record prepared particle masses and topology bonds in
the partition report. Analysis reuses that mapping. Otherwise supply `masses`
(an ordered dalton array) or `system_xml` (prepared System XML), and `bonds`
(zero-based particle pairs) for periodic reconstruction. Elemental masses are
not substituted for simulation masses; zero-mass virtual sites are retained.
DCD coordinates retain their original float32 Angstrom precision.

`--frames frames.npz` instead accepts explicit `steps`, `time_ps`,
`positions_nm` of shape `(frames, particles, 3)`, and `masses`, with optional
`boxes_nm` and `bonds`. Alternatively provide `points` of shape
`(frames, coordinates)` for externally evaluated coordinates. Step/time alignment
remains mandatory. All arrays must be readable with `allow_pickle=False`.

## Coordinate choices

All indices are zero-based prepared-particle indices, independent of boost
selection. Every coordinate requires an explicit `pbc` policy, either `none`
(already chosen coordinate images) or `minimum_image`.

- `distance`: two COM groups, output nm.
- `angle`: three COM groups, output degrees, angle at the second group.
- `dihedral`: four COM groups, output degrees in `[-180, 180)`.
- `rmsd`: `reference_nm`, `measure_indices`, explicit `align` boolean and,
  when aligning, `fit_indices`. Optional `reference_measure_indices` and
  `reference_fit_indices` specify atom correspondence in a different reference
  order. RMSD is unweighted in nm; COM coordinates use simulation masses.

The positive-mass particles in each periodic COM group must be connected by
supplied topology bonds; zero-mass particles do not contribute to the COM or
impose connectivity requirements. Molecules are reconstructed along those
bonds before mass averaging, and inter-COM
vectors use the nearest lattice image, including triclinic boxes. Ill-conditioned
boxes and undefined angles/torsions are rejected. Explicit `none` never silently
unwraps molecules.

RMSD uses a proper Kabsch rotation and translation; fitting requires at least
three non-collinear atoms. Fitting and measurement selections may differ, for
example protein fitting and ligand measurement. With periodic RMSD, fitting
must belong to one connected molecule, which retains its reconstructed image.
Each other measurement molecule is translated to its image nearest the fitting
centroid before alignment. Supply a whole
reference in the same image convention. Use `align: false` to measure in the
chosen reference frame without removing rigid-body motion.

Multiple coordinate specs produce a joint PMF. Supply one edge array and,
optionally, one boolean in `periodic` per coordinate. A periodic axis wraps into
its entire edge range; use a 360-degree range for dihedrals. Nonperiodic frames
outside the edges are counted in `n_outside`. Unequal bin volumes are included,
so the result is a probability-density PMF, with minimum populated free energy
set to zero.

For exchange trajectories every coordinate must declare `identity: "slot"` or
`identity: "physical"`. Physical identity uses the producer's recorded `exchange.copies` as
`ligand_slots`, one corresponding particle list per initially numbered ligand
slot; an explicitly supplied list must agree. The saved `slot_to_identity` permutation maps those initial identities
into current slots. Reader validation replays the completed exchange ledger,
checks configuration provenance and rejects intersecting pending transactions.
Exchange schema 2 stores summary metadata in `ligand_exchange.json` and actual
swaps in `ligand_exchange_events.jsonl`. Both files are required, including an
empty event log when no swaps occurred. Missing or incomplete event records,
inconsistent counts, and incorrect final identities are rejected. No-op checks
are omitted; identities remain unchanged between recorded swaps.
Precomputed points require `precomputed_identity` instead.
Deterministic original-style exchange is experimental: equilibrium CE2 is
rejected by default. `--allow-experimental-exchange` permits diagnostics only,
marks the output, and does not qualify the result as an equilibrium PMF.

## Results and interpretation

The NPZ contains `fes`, `biased_fes`, `direct_fes`, `counts`, `sparse`,
`mean_boost`, `variance_boost`, `log_weight`, and `edges_0`, `edges_1`, etc.
Sparse/empty PMF bins are infinity. Unavailable diagnostics are NaN.
`direct_fes` uses stable exponential averaging as a comparison, not a more
reliable default estimator. Inspect skewness, excess kurtosis and
`anharmonicity` (histogram KL divergence to Gaussian interval probabilities;
binning-sensitive, not the authors' differential-entropy estimator).

`block_stderr` uses contiguous blocks and the same reference bin in every
block. It is available only for bins populated in all blocks (`block_counts`).
Blocks must exceed correlation times; compare several block lengths before
interpreting SEM as uncertainty. `convergence_fes` contains cumulative-block
PMFs. Independent replicas and long-time convergence remain scientific
qualification work; short fixture tests do not establish convergence.

Exact analyzed `steps`, `time_ps`, `points`, channel `boosts`, identities and
`metadata_json` accompany the results for audit and later estimators. No
regression reweighting or kinetic correction is implemented here.

The deterministic fixture in `tests/v2/fixtures/gamd_pyreweighting.json` was
compared with the authors' unmodified `reweight_CE(fit=False)` function from
[PyReweighting](https://github.com/MiaoLab20/pyreweighting/blob/main/PyReweighting-1D.py).
It records the downloaded source SHA256, compatibility alias for NumPy's removed
`np.int`, and reference outputs. The comparison matches thermal energy explicitly
because the reference uses rounded `R = 0.001987 kcal/(mol K)`; runtime analysis
uses the constant above. This tests estimator arithmetic, not sampling validity.
