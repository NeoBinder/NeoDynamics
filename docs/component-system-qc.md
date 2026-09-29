# Independent component System QC

`verify_component_system_qc` reopens a `ComponentAssembly` from exact bytes and
checks the assembled OpenMM System against separately retained force-field
resources. A caller supplies the trusted SHA-256 of the canonical assembly
manifest, its recipe, reviewed states and instances, assigned parameter
packages and exact asset bytes. A GAFF ligand reference uses ff14SB protein,
TIP3P water/ions, GAFF base and the package's ligand template XML. An OpenFF
Sage ligand reference uses ff14SB, TIP3P and its generated Sage template;
the exact OFFXML hash is bound through the package's resource receipt. A
protein-only assembly uses ff14SB and TIP3P. The entrypoint
checks each XML digest against assigned package evidence before constructing
a new System through `openmm.app.ForceField`.
The [SEP peptide profile](component-sep.md) instead uses the pinned full
`ff14SB.xml`, `phosaa14SB.xml` patch and TIP3P resource. Its separate
resource order and hashes appear in the closed QC receipt.

The versioned internal receipt schema is
`contracts/component-system-qc/v1/report.schema.json`.
The canonical receipt has `numerically_executable` status only after every
particle mass, charge and Lennard Jones parameter, bond, angle, torsion,
exception, constraint, periodic box vector and supported force setting agrees.
Each component is also compared to its immutable force asset. The reference
System uses the same final topology and coordinates but a separate parameter
construction path. OpenMM CPU evaluates finite potential energy and every
force component without integration.

An imported JZ4 package can pass this check in a mixed assembly when an
assigned generated JZ4 package supplies the exact retained GAFF/template
reference and both copies' stored force terms agree. An imported-only package
without independently bound reference XML cannot use this direct-reference
profile. Numerical agreement of a single imported snapshot is not a general
qualification of imported parameter methods.

The numerical limits were fixed before comparison: force-field scalar terms
use `1e-9 + 1e-11 × |reference|` in their stated units; box vectors use
`1e-12 nm`; energy uses `1e-3 kJ/mol + 1e-7 × |reference|`; every force
component uses `1e-2 kJ/mol/nm + 1e-6 × |reference|`. Partial-charge sums
must differ from the reviewed integral formal charge by at most `1e-6 e`.
The receipt records measured maxima, exact reference resource hashes, CPU
platform with one thread and all tolerances. Package charge strings remain
unchanged; the observed Decimal residual is recorded per instance.

`read_component_system_qc_report(report_bytes, expected_sha256=...)` validates
canonical JSON, the full v1 schema, exact order-to-hash resource-role pairing,
numeric consistency and exact receipt
bytes. The caller must authenticate `expected_sha256` and bind the receipt's
`assembly_manifest_sha256` to the assembly being used. A receipt attests this
one prepared snapshot and does not qualify another ligand state, parameter
method or source pose. No execution or downstream export follows a rejected
receipt.
