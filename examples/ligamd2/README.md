# Ordered GaMD / LiGaMD2 development examples

The runnable `peptide-smoke.yaml` uses the tracked alanine-dipeptide fixture.
From the repository root, run:

```bash
neomd run examples/ligamd2/peptide-smoke.yaml --platform cpu
```

It exercises explicit selection, optional remainder, upper/lower calibration,
adaptation, frozen production and exact-frame reporting. It is a software smoke,
not a protein–ligand model or a scientific reproduction.

`single-ligand.yaml` is a prepared protein–small-molecule template. Replace the
input paths and every index list using the prepared topology before execution.
Indices are zero-based **System particles**, not PDB serial numbers. Preserve
virtual-site correspondence when translating topology indices. The resolved
lists are visible in the plan and partition provenance; inspect the partition
report before a long simulation. No pocket or ligand is inferred.

This template selects the complete bond/angle/torsion/nonbonded term when any
listed atom participates. Pocket boundary bonded terms therefore differ from
the LiGaMD2 paper's internal-only pocket partition. The 9/6 kcal/mol sigma0 pair
is an illustrative published pair, not a universal or automatically qualified
choice. Its values convert to 37.656/25.104 kJ/mol internally. Durations here are
examples, not the paper's replicated microsecond validation protocol.

A single explicit boost is valid: remove the final `remaining` entry to keep all
other interactions unboosted. Multiple explicit channels are processed in list
order; first match wins. Restraints are excluded from all explicit/remainder
boosts and remain part of the sampled potential and NPT acceptance energy.

Optional ligand exchange uses the method's explicit `exchange` configuration
and an identity ledger. The deterministic nearest-ligand extension is
experimental; conserved base/kinetic energy does not establish a Boltzmann
ensemble for its biased dynamics. Equilibrium PMF claims need separate
qualification. Use single-ligand fixed production for qualified CE2 mathematics.

See [method documentation](../../docs/methods/gamd.md) and
[the accepted scientific contract](../../docs/adr/0009-ordered-potential-boosts.md).

Run the fully generated two-copy mechanical fixture with:

```bash
OPENMM_CPU_THREADS=1 python examples/ligamd2/exchange_smoke.py
```

It writes its complete topology, explicit two-site toy-model parameters and
resolved plan before running. Both ligand copies satisfy identical parameters
and constraints; the recorded 0.7 nm distance condition is checked every step.
This is deliberately a synthetic mechanical fixture, not a parameterization
recipe for real ligands. The output is marked experimental and the standard
CE2 command refuses to interpret it as an equilibrium PMF.
