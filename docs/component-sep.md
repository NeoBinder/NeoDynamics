# SEP in a reviewed peptide

`neomd.component_sep.build_sep_protein_package` parameterizes one complete,
explicit-hydrogen peptide containing one **internal**, doubly deprotonated SEP.
The first qualified case is the curated 2MZ2 peptide: 17 residues, 288 atoms,
291 source bonds, and one SEP with 14 atoms and charge −2. The whole peptide
has charge zero. SEP stays inside one protein component so both peptide bonds
remain part of the same force field. This method does not substitute SER for
SEP or silently add missing bonds, atoms, caps, or phosphate hydrogens.

The pinned method uses OpenMM 8.6.0 and OpenMMForceFields 0.16.0 with the
installed full `amber/ff14SB.xml` (SHA-256
`f61e180832a8c6c587b797281ad0b43945962358707ebc0a4b1ce411fb3826c4`),
`amber/phosaa14SB.xml` (SHA-256
`991d13246deebe35e362ad8d99c589a4a2fcf7322c97a408578f7c8c874ae08b`),
and `amber/tip3p_standard.xml` (SHA-256
`fc528100bfb3587f632c0492166026ed5658c897026de0408f371604a79271fc`).
The phosphate file is a patch for the **full** ff14SB XML; the standard
component protein resource `protein.ff14SB.xml` is a different family and
cannot replace it. The package identifies
`openmm.ff14sb.phosaa14sb.sep` and records all three resource hashes.

## Reviewed source and selection

The private `component_generate` method is
`amber-ff14sb-phosaa14sb-sep`. Its exact options are `method_id`,
`source_residue_id`, `template_id: "SEP"`, and
`template_resource_sha256` set to the pinned phosphate XML digest above.
`source_residue_id` is the ID in the canonical source's `residues[]`, not a
sequence number. Inputs are the exact reviewed `source.json.gz` blob and
`state.json` plus `instance.json` from the same successful, owner-scoped
Review operation. The instance must bind **every** peptide source atom and
the sole source frame; the accepted state must retain all explicit hydrogens,
source bonds, and coordinate stereochemistry.

The selected residue needs the exact 14 SEP atom names and 13 internal bonds
from the installed template. P–O1P and C–O have double order; other SEP
internal bonds and both peptide junctions have single order in the accepted
state. Any explicit source order must agree. SEP N bonds to the predecessor C,
and SEP C to the successor N; no other external SEP bond is accepted. O2P and O3P have
formal charge −1 each, all other SEP atoms have zero formal charge, and CA
has S stereochemistry in both reviewed state and source pose. Inspection
returns bounded diagnostics with a code, source JSON Pointer, and atom ID/name
detail. A missing bond or changed source revision requires a newly reviewed,
explicit source revision; parameterization never repairs it implicitly.

The curated 2MZ2 PDB is the authority for the retained 288-atom pose and
hydrogen inventory, including HIE7. Deposited 2MZ2 mmCIF supports SEP
identity and both peptide links, but its model 1 has 285 atoms and differs in
some hydrogens. The curated source graph explicitly adds the 13 SEP template
bonds and one missing ARG1–SEP2 peptide bond using deposited mmCIF bond
records, with every source atom ID and coordinate retained. The original
partial graph remains available for audit.

## Package, assembly, and limits

The generator stores a full-chain force asset and exact source, instance,
template selection, atom map, three XML resources, and QC report as
hash-bound package assets. The package's `integrity_passed` result verifies
that one accepted state and parameter set; it does not itself qualify a
prepared System. Assembly uses the stored terms with TIP3P, PME at 1.0 nm,
HBonds, 4 amu hydrogen mass repartitioning, and no switching. The independent
System QC reconstructs a fresh full-ff14SB-plus-phosaa reference, compares
terms and CPU energy/forces, then gates a prepared-bundle/v2 export.

The first profile accepts one SEP peptide package only. A mixed SEP and
ligand package recipe has no qualified reference and is rejected. A terminal
SEP, S1P (the separately protonated phosphate template), a different cap or
protein family, missing junction, or different resource bytes is unavailable.
The catalog advertises the method only when exact installed resource bytes
and the qualification record match. The bounded qualification digest is
`35f9c424de558cc7aeceebe9177eacded818aa31adbf5c679fa440fa37ed536d`.
It records an independent AmberTools LEaP comparison of the full peptide and
both SEP junctions, plus same-method package and assembled-System QC; it is
not a claim about arbitrary modified residues or arbitrary SEP contexts.

The phosphate/Amber-derived resource bytes are loaded from the pinned
installation and retained only in task-local evidence. Redistribution terms
for those files need a separate release determination before copying them
into an image or published fixture. See the
[OpenMMForceFields source](https://github.com/openmm/openmmforcefields) for
the installed template family.
