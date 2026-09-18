# Protein preparation preset catalog

`catalog.schema.json` records accepted parameter defaults and exact installed
forcefield resource provenance. This foundation has no preparation request,
compile, upload or execution route yet. The inventoried assembly choices are
confirmed and `unresolved_parameters` is empty. `system_assembly_available`
stays false until actual preparation/lineage implementation and qualification;
resolved metadata alone does not enable execution.

The accepted 0.15 mol/L NaCl concentration is a user-editable default, alongside
padding and pH. Neutralizing ions are additional to the configured salt amount.
Existing water is retained before completing the water box; the catalog names
this accepted behavior `retain_and_solvate`. This is not yet an executed
preparation operation.
The HMR parameter is explicitly 4 amu, constraints use HBonds and water is rigid.
Center-of-mass motion removal is explicitly disabled (`remove_cm_motion=false`),
and the nonbonded method is PME (`nonbonded_method=pme`), with a 1.0 nm cutoff
and Ewald tolerance 0.0005. Dispersion correction is enabled; switching is disabled
(`switch_distance_nm=null`), periodic exceptions are disabled, and constrained
degrees of freedom omit harmonic parameters (`flexible_constraints=false`).
The scientific protocol schema separately records the confirmed 2 fs initial
editor timestep; it is not a preparation catalog field. The catalog does not
claim a final System particle-mass inventory.

`preset_revision_sha256` hashes the preset without that field using UTF-8
sorted-key compact JSON, no nonfinite values and no trailing newline. It is not
the registered scientific protocol hash. Each resource descriptor binds exact
bytes and package version. Resource IDs are catalog identifiers, not caller
paths or permission to load additional XML.
