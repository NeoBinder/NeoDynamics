# Component preparation scientific artifacts v1

These seven self-contained Draft 2020-12 schemas define immutable scientific
artifacts and bounded manifests. Every top-level document has its exact
`urn:neodyn:component-preparation:v1:<kind>` `schema_id` and integer
`schema_version: 1`. Consumers reject unknown versions before mutation; they
never fetch a remote schema or reinterpret an older artifact. This contract
does not replace molecular-topology v1 or prepared-bundle v2.

| Document | Storage boundary |
| --- | --- |
| `chemical-state` | Full reviewed atom/bond graph as a separate artifact, up to 256 MiB decoded; never inline a public control request. |
| `component-instance` | One copy's source membership, atom bindings and pose frame as a separate artifact, up to 256 MiB decoded. Multiple copies may share one state/package but keep different instance IDs and maps. |
| `parameter-package` | At most 1 MiB manifest; charge/term arrays, parameter templates, the parameter atom map and contextual attachment terms are referenced assets. |
| `preparation-recipe` | At most 1 MiB immutable assignment and assembly manifest with exact source, state, instance and package identities. |
| `capability-catalog` | Static scientific qualification scope, at most 128 methods. |
| `diagnostic` | Bounded code, RFC 6901 path and safe detail. |
| `artifact-reference` | Owner-scoped opaque ID, kind, encoded SHA-256, encoded/decoded lengths, encoding and media type. |

Artifact references contain no path, URL, signed URL or executable content.
`content_sha256` covers the **exact encoded transfer bytes** and is verified
before bounded gzip decompression. A linked document's `state_sha256`,
`instance_sha256` or `package_sha256` covers **RFC 8785 canonical decoded JSON**.
Identity encoding requires equal lengths; gzip requires `application/gzip` and
one complete member with no trailing bytes. Source topology uses its existing
`urn:neodyn:molecular-topology:v1` identity and 64 MiB encoded / 256 MiB
decoded limits. Control JSON is limited to 1 MiB. Parameter assets are at most
512 MiB encoded and decoded; the package manifest stays small.

JSON Schema validates fields and bounds. `neomd.component_contract` additionally
validates canonical u64 atom IDs, sorted unique atoms/bonds, formal charge sums,
accepted stereo/protonation decisions, one-to-one instance bindings, package
charge sums, recipe links and RFC 6901 escapes. A complete source map check
requires the authoritative molecular-topology v1 membership for each instance;
the caller supplies it to `validate_component_set`. Generated hydrogen atoms
without source counterparts are recorded later in preparation lineage.
The recipe's required `assembly.solvation_seed` is a 32-bit unsigned integer.
It binds stochastic solvent and ion placement to the accepted recipe; the
assembly report records the actual seed and algorithm/runtime identity.

The package manifest's charge sum is checked against the state's net charge.
The referenced parameter asset must separately be decoded and checked for
atom-by-atom charge/term coverage before scientific use; a manifest or schema
pass alone is no such check. Contextual covalent packages require explicit
attachment sites and cross-boundary term assets. Assemblers consume accepted
packages and cannot silently regenerate them.

Catalog `qualified` means exact scientific source, resource, platform,
chemistry and receipt evidence. Public executability also requires a fresh
API-owned executor observation. Registration, an installed executable or a
schema-valid fixture cannot mark a method available. New component methods and
imports start unavailable until separately measured and qualified.

The positive and adversarial fixtures are **synthetic contract examples**, not
scientific or api-dev qualification. Their only catalog method is unavailable.
`build_schemas.py` and `build_fixtures.py` reproduce the tracked JSON; the latter
uses RFC 8785 for cross-document hashes. Existing stored artifacts and active
sessions remain unchanged; a future format migration must be explicit and
versioned.
