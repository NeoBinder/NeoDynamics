# Preparing and inspecting a system

## CLI preparation

`neomd prepare prep.yaml` (or `prepare_system(config)`) builds
protein + ligand + solvent systems and writes `solv.pdbx`, `system.xml` and
`ligand.json`. External binaries are wrapped by subprocess-isolated adapters
in `src/neomd/tools/`: `antechamber` (GAFF parameterisation), `orca` + Multiwfn (RESP2
charge fitting), `ligand` (RDKit/openff processing), `convert`, `fix_protein`
(PDBFixer) and `template_xml`. All OpenMM private-API usage lives in
`openmm_privates.py` behind a pinned-version gate (openmm 8.6.x) that raises
`UpstreamVersionError` otherwise.

## Inspecting force-field parameters

To inspect the prepared force field, `bin/inspect_ff_params.py` reads
`solv.pdbx` + `system.xml` and prints parameter tables for a user-selected
atom set. Selection keywords (`chain`, `resid`, `resname`, `name`,
`index`, `resindex`, `element`, `distance`) form a small expression
language: `;` / `and` / `&` are AND, `or` / `|` are OR, `not (...)` /
`!(...)` negate, `,` lists values of one keyword, and
`distance=5:(...)` keeps atoms within 5 angstrom of a parenthesised
sub-selection (precedence NOT > AND > OR) — e.g.
`chain=A and (element=C,N,O or resname=JZ4) and not (name=H)`. Repeated
`--select` groups are OR-ed. `--params` picks from the types actually
present in the XML (bonds/angles/dihedrals/nonbonded/mass) and
`--mode any|all` decides whether a bonded term is shown when it touches
the selection or only when it sits fully inside it. The same logic is
importable as `inspect_params(...)` for Python use.

## SDK preparation and verification

Use [protein preparation](protein-preparation.md) for source identities,
grouping, evidence-gated repair, retained hydrogens and waters, and explicit
System assembly. [Prepared bundles](prepared-bundle.md) explain UUID/particle
correspondence and independent verification. These SDK paths have different
contracts from the CLI workflow; see their supported-history limits before
choosing one.

For a complete protein–ligand walkthrough, use the
[3HTB example](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/3HTB_complex).
Service intake and remote operation limits are in the
[resident service guide](service-runtime.md).
