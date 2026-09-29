# Component chemistry review

`neomd.component_inspection.inspect_components` partitions a canonical molecular-topology v1 revision by its stored chemical bonds. It keeps each connected copy and source frame separate; residue boundaries do not define molecules. The result binds the topology ID, revision ID and canonical JSON SHA-256. Partial connectivity, unsupported bond kinds, missing ligand charge/order, coordination and declared but detached covalent attachments produce typed diagnostics. Inspection does not infer bonds from coordinates.

`review_component(records, state, instance, evidence_bytes=...)` checks one accepted component-preparation v1 `ChemicalState` and `ComponentInstance` against that source. Bindings must exhaust exactly one connected source unit. Known source elements, formal charges, bond orders and stereo cannot be overwritten by a state; edit and review a new source revision first. Generated state atoms may only be hydrogens, each with one recorded source parent. The review checks local stereochemical feasibility and compares declared R/S centers with independently assigned CIP from the selected source pose where enough heavy-atom coordinates exist. It returns canonical immutable state and instance bytes plus a source-linked change report. The host must authorize evidence attachments and preserve those exact bytes; a SHA alone does not authenticate a user decision.

The state contains chemistry but no bound coordinates. Each instance binds one source frame and its own source-to-state atom map. Copies may share the same state and parameter package while retaining distinct poses. `generated_package_cache_key` includes the state, exact method/charge backend, tool and resource versions, options (including charge choices), parameterization conformer and seed. It does not include the bound instance pose.

`neomd.component_protein.inspect_protein_variants` checks explicit ff14SB residue variant assignments in the full stored-bond complex against pinned heavy and hydrogen template inventories. It reports missing qualified repair, hydrogen conflicts and external-bond mismatches without selecting a variant from pH. Missing hydrogens are reported for a later qualified addition step; the inspection itself does not prepare a System.

These interfaces establish chemical review and structural preflight only. They do not make a ligand method or covalent template scientifically qualified. Existing complete-topology preparation uses the same component grouping for ligand molecules and continues to require its separately qualified source and force-field path.

The `prepare_system` coordinate-file route also rejects an inter-residue
covalent link before writing prepared artifacts when it is outside ordinary
peptide or cysteine disulfide connectivity. Its error identifies both source
residues and atoms. In particular, a Ser195–1ZG attached 1CGH pose cannot be
prepared by treating 1ZG as a separate ligand: that attached group needs a
qualified joint parameter template. A fully parameterized external AMBER or
GROMACS topology uses its separate import route.
