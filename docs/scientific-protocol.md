# Scientific protocol compilation

The independent `neomd_server.protocols` module converts closed scientific
protocol v1 JSON into an immutable engine `Plan`. It accepts the currently
qualified MD, minimization, WT, OPES, SMD, GaMD and full-ladder RBFE settings,
resolving ordered prepared atom UUIDs into System indices. The engine and CLI
retain their existing Plan interfaces and do not import the server package.

The caller supplies an authorized prepared manifest, its immutable artifact
directory, trusted protocol/output directories and explicit `ScientificLimits`.
`compile_protocol` returns `CompiledProtocol` with the Plan, canonical protocol
SHA256, CV/restraint/steering UUID-to-engine-name maps and computed resource
requirements. `validate_protocol` performs the standalone closed JSON-schema
check; file and scientific correspondence checks occur during compilation.
Both expose collected `ContractError.problems` with field paths.
The confirmed initial timestep for future new-protocol editors is 2 fs
(0.002 ps), recorded in the schema's `physics.timestep_ps.default` annotation.
Every registered protocol must still contain an explicit timestep. Validation
never inserts it and compilation preserves the submitted value; existing
protocols and CLI defaults are unaffected. Temperature, friction and other
unresolved scientific choices are not supplied by this authoring annotation.

```python
from neomd_server.protocols import ScientificLimits, compile_protocol

# All directories, binding and limits are supplied by the trusted host.
compiled = compile_protocol(
    descriptor,
    prepared_manifest=verified_manifest,
    prepared_directory=prepared_directory,
    output_directory=output_directory,
    protocol_directory=protocol_directory,
    limits=ScientificLimits(
        max_bias_grid_points=host_grid_limit,
        max_reference_frames=host_frame_limit,
        max_reference_bytes=host_reference_byte_limit,
    ),
    reference_resolver=resolve_owned_coordinate_reference,
)
```

RMSD and path CVs use `prepared_start` or an immutable registered reference UUID.
The resolver returns `CoordinateReference(metadata, path)` only after ownership
checks. Metadata binds the prepared revision, complete ordered particle
identities, exact bytes, float64 little-endian nm representation and frame count.
Large references stay as files; protocol JSON carries no coordinate arrays or
paths. The compiler verifies and writes full-System-row reference PDBx files
under its trusted protocol directory. The simulation retains its original
chemical topology; the host injects complete N-row starting coordinates and the
explicit topology mapping via `KernelSpec.initial_positions_nm` and
`topology_particle_indices`.

The language-neutral schemas and examples are distributed in
`contracts/scientific-protocol/v1`. Their README defines units, native coefficient
conventions, ordered CV axes, SMD ramp behavior and the reference hash. No
arbitrary Plan, output path, plugin, ML model or expression is accepted. GaMD
custom force-group channels are unavailable; RBFE requires an already
alchemified prepared System. A protocol does not confer ownership or certify
physical validity.

Prepared artifact intake first checks schema/hash consistency and approves its
XML force/type vocabulary. It can then call the generic
`neomd.system.verify_prepared_system(manifest_path, expected_sha256=...)`, which
returns the manifest after comparing actual particle counts, masses, virtual
site definitions and positions, topology row/membership/bond correspondence,
force inventory, constraint count and periodic box. The verifier uses public
OpenMM preparation APIs, with no service import. PDB/PDBx topology coordinates
are compared within their writer's decimal quantization; full float64 particle
coordinates preserve independent exact artifact hashes. Bond endpoints are
verified; serialized topology formats do not certify retained bond order/type
annotations. XML correspondence does not replace force-field/QC assessment or
successful Context construction.

The public test surface includes real CPU completion of all nine method names,
all CV and restraint translations, full particle references with a virtual site,
native angular energy units, rejected unsafe fields/unknown identities,
reference revision/order/hash/units, explicit resource guards and self-consistently
rehashed malicious System changes. These tests qualify the compiler and generic
scientific verifier, not service admission, recovery, deployment or force fields.
