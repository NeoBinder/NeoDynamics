# 1crn.pdbx provenance

- Source: RCSB Protein Data Bank entry 1CRN (crambin, 46 residues, 0.83 Å),
  official PDBx/mmCIF file retrieved from
  https://files.rcsb.org/download/1CRN.cif on 2026-09-21.
- License: wwPDB data files are distributed under CC0 1.0 (public domain
  dedication); https://www.wwpdb.org/about/usage-privacy-policy.
- Original file SHA256:
  23787562c427d7c1abe5420e86d5f1d0a6c7007dec1e8ce85645a6d69c32e8ba
- Normalized fixture SHA256:
  d3115032db8e0a6c9c4fff1d0259b531c70a22593a6aa1309528992ad9f663e4

## Normalization

The official file embeds the full `chem_comp_bond` dictionary (262 rows).
OpenMM's PDBxFile reader turns those records into typed bonds
(order/Single/Double), while its writer never emits `chem_comp_bond` and
re-derives standard-residue bonds typeless from `residues.xml` templates, so
the assembly export round-trip gate rejects such sources. The desktop capture
profile emits `atom_site` plus `struct_conn` (disulf) only, so the fixture
strips exactly the `chem_comp_bond` loop and nothing else; all `atom_site`
rows and the three disulfide `struct_conn` records are byte-identical to the
original. Characteristics: 327 atoms, 46 residues (all standard), 1 chain,
337 loaded bonds including 3 disulfides, no waters, no hydrogens.

Qualified 2026-09-21 through the real SDK path (load+descriptor validate,
prepare, independent verify): wall 11.4 s, 7473 particles, CPU, seed 42.
