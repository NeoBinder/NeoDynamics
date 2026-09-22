# Trusted runtime qualification fixture

Six argon atoms with harmonic bonded interactions, a torsion term, and an actual
`lambda_sterics` global parameter. Generated locally through public OpenMM 8.6
and `export_system_bundle` by `tests/server/generate_qualification_fixture.py`.
The source contains only synthetic coordinates and generated UUIDs.

Pinned manifest SHA-256:
`c25b295179d0bee7157e42d92945c3a4917b08883eadbfaf44caa7d736061623`.
That manifest records the hashes and sizes of every scientific file. The probe
also verifies actual prepared-system semantics before making any Context.

This fixture qualifies execution/control plumbing, not protein force-field
accuracy, RBFE physical convergence or production performance. Regeneration
creates new UUIDs; review all artifact changes and update the pin explicitly.
