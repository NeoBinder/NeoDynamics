# Canonical molecular topology transfer v1

The gzip member contains RFC 8785 canonical JSON matching `topology.schema.json`.
The compressed SHA256, compressed byte count, decoded byte count and topology
revision are part of the immutable reference. The SDK enforces 64 MiB compressed
and 256 MiB decoded limits before scientific use. The host must resolve the
reference through its owner-scoped upload store; the reference has no file path.

Bond records are the only chemical edges at this boundary. PDB/mmCIF producers
may have materialized bonds from a pinned standard-residue template during import;
those rows carry `template_reconstruction` provenance and the exact template
reference. The scientific adapter never runs a template. Endpoint pairs are
canonical and unique. A coordinate frame maps positions by atom ID and declares nm or angstrom;
the adapter converts angstrom values to nm for OpenMM. Coordination and other
relations are retained in the source records and never added as OpenMM bonds.
OpenMM requires every atom to have a residue and chain, so the adapter creates
private containers for parentless rows while retaining original membership in
the canonical document. It does not parse PDB/PDBx or call
`createStandardBonds`.

| Capability | Coverage | Qualification |
| --- | --- | --- |
| Validate and view records, coordinates and typed relations | complete, partial or unknown | Bounded canonical decoder and real OpenMM graph adapter |
| Construct a harmonic-bond System from explicitly supplied masses and per-bond length/stiffness | complete | Local CPU energy evaluation; no force-field or protein preparation claim |
| ff14SB System construction for the reviewed, complete Ala2 artifact | locally qualified | `ala2-qualified.json.gz` supplies 22 atoms and 21 stored pairs; a real CPU OpenMM System and finite energy use the record adapter without a PDB reader |
| Local ff14SB/TIP3P/GAFF preparation of the reviewed 3HTB artifact | qualified in SDK | `prepare_explicit_topology` consumes the immutable stored graph, adds hydrogens and solvent, builds a real System and publishes a checked source-to-prepared map plus explicit bond additions |
| Service/public v2 preparation from this artifact | unavailable | The runtime provider and gateway still accept only PDBx; owner-scoped v2 upload, operation/result contracts and recovery are not implemented |

`tests/v2/test_topology_adapter.py` exercises the supplied cross-boundary bond
fixture through a real OpenMM CPU Context and checks exact atom ID mapping,
coverage admission, coordinate-unit conversion and immutable artifact validation.
The fixture's simple harmonic parameters are test inputs, not a preset for
production chemistry. The separate Ala2 test freezes the bond projection from
NeoDynamics `prepare_system`, then applies a reviewed fixture-specific complete
coverage assertion. The 3HTB example's prepared manifest likewise retains all
1,329 protein and 22 ligand source bonds between original atoms, without a
cross-component bond; generated hydrogen and solvent bonds are preparation
changes. `tests/v2/test_3htb_bond_reference.py` checks its frozen source pair
projection. `3htb-qualified.json.gz` freezes the reviewed complete source
graph and marks JZ4 as the ligand parameterization role. Its local SDK test
verifies the final prepared graph as the exact source pairs plus explicit
hydrogen and solvent additions. The ligand needs supplied formal charges,
supported bond orders and unique atom names. The current path requires an
orthorhombic source box that meets the requested solute padding; covalent
protein–ligand templates and nonchemical relations are unsupported. Neither
comparison upgrades arbitrary PDB/mmCIF imports from unknown coverage.
