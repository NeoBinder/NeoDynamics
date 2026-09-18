# Scientific protocol v1

`protocol.schema.json` (`urn:neodyn:scientific-protocol:v1`) is the closed
scientific configuration carried by the service's protocol registration
contract. It is independent of admission, identity scopes, HTTP and storage.
`coordinate-reference.schema.json` (`urn:neodyn:coordinate-reference:v1`)
describes immutable full-System-row reference coordinates. Resolve both schemas
locally; validation must never retrieve a schema over the network.

A protocol binds one `prepared_revision_id`. The authorized prepared binding
in the enclosing request also pins manifest and mapping SHA256 values. Every
selection field contains ordered, unique prepared atom UUIDs. Integer indices,
selection expressions, source atom UUIDs from another revision, file paths,
URLs, plugins, model loading, arbitrary Plan sections and output destinations
are rejected. Selection resolution retains order and requires massive,
control-eligible chemical atoms; virtual-site identities remain present in
reference coordinates but cannot be selected for these centroid controls.
Definitions use unique UUIDs. The compiler supplies safe engine names and returns
UUID-to-name maps. Metadynamics/OPES `cv_ids` explicitly defines axis order and
must reference every supplied CV exactly once.

## Methods and science

| Method | Required method settings |
| --- | --- |
| `min` | `minimization`: finite positive tolerance and bounded maximum iterations |
| `eq`, `md`, `prod` | `steps`, using the existing engine dynamics behavior |
| `metadynamics` | Ordered `cv_ids`, bias factor, hill height and deposition cadence |
| `opes` | Ordered `cv_ids`, pace and barrier; optional standard/explore mode and KDE settings |
| `smd` | `steering` definitions, with scalar or piecewise linear native ramp anchors |
| `gamd` | Total/dual mode, sigma0, calibration sampling and reporting cadence |
| `rbfe` | Distinct ordered lambda windows over declared prepared-System global parameters |

All methods take explicit temperature K, timestep ps, friction ps⁻¹ and seed.
The user confirmed 2 fs (0.002 ps) as the initial timestep for new protocol
editors. The schema's `default` is an authoring annotation: `timestep_ps` remains
required, validators do not fill it, and compilation preserves an explicit
caller value. Existing protocols and qualification fixtures keep their chosen
timesteps. This annotation does not choose the remaining physics parameters or
implement a protocol editor.
Optional pressure uses bar and requires a periodic prepared System. Reporting
cadences are steps; no reporting field changes a filesystem destination. RBFE
requires positive energy reporting. Its input System must already be alchemified;
this protocol does not construct an alchemical force field or expose mock bias.
GaMD dual mode still requires real torsion groups. Opaque custom force-group
channels, ML and plugins are outside this version's qualified vocabulary.
Pending prepared-System modifications must first be materialized and exported
as a new immutable prepared revision.

The nine CV types and ten restraint types mirror the qualified native registry.
The JSON schema lists every accepted parameter and required field. Coordinates,
reference positions, distance grids and path smoothing length are nm; CV angle
bounds/widths are degrees; coordination and path progress grids are dimensionless.
Energies, WT heights, OPES barrier and GaMD sigma0 are kJ/mol. Path reference
frames use the existing optimal-alignment RMSD definition.

Restraint numeric coefficients preserve the native expressions. Distance,
reference-position and Cartesian walls act on nm deviations to `order` (default
2); vector/RMSD/funnel walls use their native quadratic expressions. Angular
bounds are written in degrees and converted to radians inside OpenMM; the bare
`restr_k` is **not** rescaled as a per-degree spring constant. Dihedral keeps
its native wrapped atan/tan expression, including wrapped intervals such as
150 to −150 degrees. Boresch uses `restr_k_r` in kJ/mol/nm²,
`restr_k_theta` in kJ/mol/rad², and `restr_k_phi` in kJ/mol for its native cosine
term. Funnel `width`, `steepness`, `s_center` and `buffer` retain the native
numeric parameter convention, including its existing sigmoid expression.
`restr_k_per_atom` explicitly multiplies the selected atom count; exactly one
of it and `restr_k` is accepted for reference-position restraints.
Zero walls remain explicit. `distances` has per-pair `order`; a top-level
`order` is rejected because the native packed force reads per-pair values.

SMD accepts native ramp fields only. A bare xyz triple remains one position;
a list of xyz triples is a position ramp. One-element lists are fixed values.
There must be at least one step per ramp segment. The engine retains its
floor-sized segment schedule and 5000-step update cadence; a short run can
therefore retain the first applied value throughout. Static and steering UUIDs
cannot collide. Compiling does not change a method's equations or defaults.

## Coordinate references

`{"kind":"prepared_start"}` takes the prepared bundle's complete N×3
`coordinates.f64le`. `{"kind":"registered","reference_id":"…"}` resolves
an immutable artifact through the authorized host. RMSD requires an explicit
`frame_index` when that artifact has multiple frames. Path CVs use all frames,
require at least two and reject `frame_index`.

Reference bytes are frame-major, then System particle index, then xyz: exactly
`frame_count * particle_count * 3 * 8` bytes of little-endian float64 nm.
`particle_identity_sha256` hashes RFC 8785/JCS bytes of the ordered JSON array
of all `manifest.particles[*].identity_id` strings, including extra virtual sites.
The reference metadata pins revision UUID, identity order, exact content hash,
counts, units and representation. The compiler checks these bindings, finite
coordinates and host resource limits before writing full-System-row coordinate
reference PDBx files in its host-owned protocol directory. Those files are
reference carriers, not replacement chemical topologies.

The host must provide `ScientificLimits(max_bias_grid_points,
max_reference_frames, max_reference_bytes)` explicitly. No deployment values
are supplied here. The compiler checks joint grid product, frames per reference
and aggregate reference bytes, then returns the computed requirements for host
admission/capabilities. References count per use, including the complete
registered artifact when RMSD selects one frame, bounding verification input
cost as well as later materialization. Structural JSON bounds do not constitute
runtime admission or a promise that a particular device can execute the plan.

## Fixtures and qualification

`fixtures/water-{md,metadynamics,path}.json` bind the real synthetic
`prepared-bundle/v1/fixtures/water-extra-site` revision. The two-frame
`water-reference.f64le` and metadata include the fourth virtual particle.
These are compiler/contract fixtures, not a scientific production protocol.

Schema and byte-hash verification establish representation integrity.
`neomd.system.verify_prepared_system` additionally compares actual deserialized
System inventory with the manifest after the host approves XML types. Neither
step certifies a force field, chemical appropriateness, stability, QC approval
or device compatibility. Context construction and execution qualification
remain required gates; authorization and immutable artifact lifetime belong to
the host.
