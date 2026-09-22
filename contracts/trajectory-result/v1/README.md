# Trajectory result manifest v1

`manifest.schema.json` (`urn:neodyn:trajectory-result:v1`) defines producer-owned
evidence for closed DCD segments. See `docs/trajectory-results.md` in the owning
repository for finalization and trust semantics. This contract is separate from
remote transport, publication and storage policy.

All objects are closed. IDs are canonical UUID strings and SHA-256 values are
lowercase hexadecimal. Hash material uses RFC 8785 canonical JSON (JCS):

- Particle order: `SHA256(JCS(ordered manifest.particles identity_id strings))`,
  including every atom/site row.
- Protocol binding: `SHA256(JCS({prepared, protocol}))`, where `prepared` is the
  complete binding and `protocol` is the registered scientific descriptor.
- Artifact hashes cover exact file bytes; no ETag, filename or path is an identity.

The schema does not replace semantic checks. Consumers must reject repeated
artifact IDs/series, invalid window ordinals, inconsistent frame-count/last-step
arithmetic, a different prepared/protocol/execution binding, or changed bytes.
Frame time is local step multiplied by the encoded timestep in ps; RBFE windows
are independent physical series. A terminal execution step may be later than the
last reported frame. Empty trajectories and a zero-frame trajectory are distinct.

The local artifact descriptor returned by the trusted producer binds these
manifest bytes to their service execution. A self-authored manifest is only a
pairing assertion. The JSON format contains no authenticity signature and does
not itself grant download authorization or certify arbitrary uploaded files.
