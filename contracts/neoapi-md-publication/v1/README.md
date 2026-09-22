# NeoAPI MD publication v1 consumer

This directory is an attributed, independent consumer snapshot of NeoAPI-rs
`openapi/internal/md-publication-v1.yaml`, SHA256
`7c43f5341e6effa21d8c8731418ceeb9ac87ba1d21cf3e1a14b250c21335ea22`.
NeoAPI owns this interface; it is separate from Dynamics' opposite-direction
`contracts/imd/v1`. No sibling repository is imported at runtime.

`openapi.yaml` is byte-for-byte upstream. `schema.json` copies only
`components.schemas` and rewrites local component references to `$defs`.
`snapshot.json` records provenance, every copied file hash and the frozen error
status mapping. No network schema retrieval is permitted. Refresh/check from
an explicit local upstream checkout with:

```sh
python contracts/neoapi-md-publication/v1/refresh.py /path/to/NeoAPI-rs/openapi --check
```

The three upstream OutputSet fixtures retain their exact JSON, RFC8785 JCS
bytes without newline, and SHA256. They distinguish changed generation and
prepared binding. `consumer-multipart-empty.json` is an independent synthetic
metadata fixture: one JSON companion, a 5 MiB plus 17-byte artifact and a
zero-byte object. It qualifies descriptor/geometry validation only; it is not
a scientifically valid H6 manifest or evidence of uploaded bytes.

`neomd_server.publication_contract` contains immutable validated document bytes
and explicit expected-binding decoders. JSON parsing is bounded before decode
at 4 MiB and 32 nesting levels; schema structure is much shallower. Duplicate
keys, invalid Unicode, unsafe/nonintegral numbers, unknown properties and
invalid schemas fail closed. Integral JSON numbers such as 1.0 normalize to
safe integers, matching the API's DTO semantics. UUID validation follows the
frozen canonical wire format; it adds no identity authorization rule.
Validation collects up to 64 value-free problems and marks truncation. Error
paths contain only known schema field names and numeric indices, never an
untrusted unknown property name or provider diagnostic value.

A publication expectation binds the complete canonical request plus provider
and scope, and the publication ID once it is known. Uploads must have unique
identity and correspond one-to-one to the artifact set with immutable
geometry/expiry. Empty objects have zero parts.
Part pages use the required exclusive `after_part_number`, initially 0; a
nonnull next cursor equals the last returned number. Page receipts still need
`match_part` against the producer's previously recorded exact local
upload/number/offset/length/SHA256 tuple.

A matched receipt reconstructs the original OutputSet by removing only each
artifact's opaque `download_ref`. The entire binding, complete sorted artifact
set, hash, original filename and media type must match. Complete publications
require that receipt and all uploads verified. A previous complete receipt
cannot be overwritten by an older observation. Intermediate states have no
invented total ordering. The response contract has no revision, so later
journal/transport ownership must still fence concurrent responses.
Use `PublicationSnapshot.receipt()` for the first embedded receipt. A standalone
`decode_receipt` requires the opaque receipt ID from the GET selector and can
compare a previous matched receipt's complete immutable bytes. Taking the
expected ID from the new response would not validate the requested resource.

Validated or matched bytes alone prove no authenticated provider, finalized
scientific outcome, OSS readback or permission to remove local data. This
module performs no HTTP/SDK/DB/artifact work, retry, journal update, runtime
registration or cleanup. An `Error.request_id` is only HTTP trace correlation.
A durable complete matching receipt must precede any future cleanup
eligibility. Request construction does not bless caller metadata as trusted
terminal evidence. Selected-H6 size and deployment admission budgets remain
the API's separate admission checks.

This snapshot is packaged as read-only package data (see `pyproject.toml`
`[tool.setuptools.data-files]`), so the resident service reads the frozen
contract from its installed distribution; runtime coordinator activation is
separate host configuration (`NEOMD_SERVER_PUBLICATION_*` settings).
