# Component system assembly

`assemble_component_system` consumes an accepted preparation recipe, its exact
canonical source topology artifact, reviewed states and instances, and the
referenced immutable parameter package assets. It builds one OpenMM System
without running repair, hydrogenation, charge assignment or a parameterizer.
The assembly receipt has `assembled_pending_independent_qc` status. It does
not make a package or prepared system publicly qualified.

## Input and scientific choices

The caller supplies `source_artifact_bytes`, `states`, `instances`, `packages`,
`recipe`, and `assets_by_package` to `assemble_component_system`. The keyword
`prepared_pose_coordinates_nm` maps each included instance UUID to every
state atom ID and its `[x, y, z]` position in nanometers. Source-bound
positions must equal the selected canonical source frame. A state atom absent
from that frame requires a verified stage pose receipt in the optional
`trusted_pose_receipts` map. Its value contains canonical `receipt_bytes` and
their host-trusted `trusted_sha256`. The receipt binds the exact source
artifact, state, instance, complete pose rows, builder and verified stage
digest. The host must authenticate the trusted digest and the stage's
scientific validation; assembly checks byte integrity and bindings.
For a reviewed component, source atom formal charges may be unknown when its
accepted state explicitly resolves every mapped charge. Other source chemistry
diagnostics still reject assembly; known source charges must agree with the
accepted state.

The v1 profile accepts ff14SB protein with one compatible ligand family
(GAFF2 or [OpenFF Sage 2.2.1](component-openff.md)), TIP3P water, Na+/Cl-,
HBonds, 4 amu component hydrogen mass repartitioning, PME with a 1 nm cutoff,
and no switching. All choices, including `solvation_seed`, box policy, water
retention, ionic strength and neutralization, come from the recipe. A source
cell is recorded separately. It becomes the simulation cell only when the
recipe explicitly chooses `reuse_source_orthorhombic` and the cell passes
the PME cutoff check. Otherwise OpenMM generates a new orthorhombic cell.

Source water is retained or discarded according to the recipe. The pinned
OpenMM 8.6 solvent placer reads solute charge and Lennard Jones values from
stored force assets. Its ion step counts retained and generated waters, but
replaces only generated waters. Final solvent and ion force terms come from
the exact pinned TIP3P resource. Every component's masses, charges, bonded
terms, constraints and nonbonded exceptions come from its stored force asset;
loading a combined protein and GAFF XML set would allow duplicate parameter
rules to change those terms.
The same core can consume a declaratively imported package when its retained
force asset and compatibility pass preflight. That assembly remains pending
independent QC; an import receipt alone does not qualify a parameter method.
GAFF and Sage ligand packages cannot be mixed in one assembly: their exact
1–4 electrostatic profiles differ, and a mixed reference has not been
qualified.
An [SEP peptide package](component-sep.md) uses the distinct full
`ff14SB.xml` plus `phosaa14SB.xml` family. Its first profile accepts the
complete peptide as one package; mixed SEP and ligand packages are rejected.
Inspection and assembly reject a covalently connected protein–ligand source
unit until a qualified joint template and its complete attachment terms exist.
A durable service-side intake rejects that source as well, before publishing a
prepared bundle. Inspection uses `capability_unavailable` for the same connected
unit.

## Reopening and QC

The returned `ComponentAssembly` contains `manifest_bytes`,
`topology_bytes`, `positions_f64le`, `system_xml`, and `report_bytes`. The
canonical manifest hashes and sizes the other four artifacts. It binds the
recipe and source hashes, ordered particle to instance/state/source atom
mapping, selected box, and any trusted pose receipt hashes. Positions are
row-major little endian float64 nanometers; the System is OpenMM XML.
The topology JSON records ordered atoms, residues with chain membership,
bonds and box vectors. The report records assignments, source changes,
solvent and ion counts, net charge, constraints, exceptions, component HMR
and virtual-site inventory. Independent numerical QC must reopen the exact
bytes and compare terms, energy and forces against a separate reference
before publication or downstream use as a qualified system.

The caller owns artifact storage, access control and acceptance of scientific
review receipts. This SDK function does not grant public capability status.

## Prepared bundle v2 handoff

`build_component_prepared_bundle_v2` accepts the five assembly byte artifacts,
the exact source topology artifact, recipe and instances, plus an independently
verified component System QC report and its host-trusted SHA256. It returns a
v2 manifest and artifact byte map. The handoff retains the five assembly
artifacts byte for byte, adds the QC report and source artifact, and provides
v2 source/particle UUID bindings. `read_prepared_bundle` can reopen the
result after a durable writer stores each named artifact and the manifest.

The adapter requires QC status `numerically_executable` and binds its report
to the assembly manifest. It rejects any source atom removed by the recipe,
including source component exclusion or discarded source water. The current
prepared-bundle/v2 source schema has no disposition for those removals; they
remain supported by the assembly core for review but cannot be exported as
a v2 handoff. Numerical QC is a system check, not a public availability or
method-wide scientific qualification grant.

## Remote-service boundary

Durable component operations — owner-scoped uploads, review decisions, package
generation, import and assembly with published receipts — belong to the separate
service repository, together with their routes, contracts and availability
catalog. The SDK exposes the synchronous builder and QC functions above; it
makes no claim about a deployed catalog.

The deposited 3QQA/TCH complex has known polymer sequence gaps. Protein
repair and `prepare_system` reject this source before publishing a System;
a coordinate-complete revised source is required.

The retained SDK 3HTB/JZ4 preparation case is an **unrelaxed** numerical snapshot.
Its close generated-water pair requires minimization before dynamics; a later
phase gate owns minimization and equilibration. A service receipt from this
operation alone does not authorize a simulation start.
