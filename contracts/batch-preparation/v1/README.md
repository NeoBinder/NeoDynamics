# Trusted preparation transport v1

`neomd batch-prepare` consumes `/inputs/request.json` and
`/inputs/source_bundle.zip`. It materializes exact frozen YAML and validated
members under `/work`, and writes `/outputs/result.json` plus
`/outputs/artifacts.zip`. Handled failures exit zero so the workflow can collect
terminal reports; callers must inspect the scientific outcome. A killed or
unstarted process has no synthesized result.

The private request is limited to 65536 raw bytes. All fields are closed and
both YAMLs use the native preparation v1 loader. Only protein.path,
ligands.<key>.path and output_dir may change, preserving typed key order,
omissions and numeric types. Input paths become /work/assets/<id>.<format>;
output_dir is /work/output. Builtin resource identifiers stay unchanged.

The canonical source manifest declares every ZIP member except manifest.json.
Archive limits: 128 entries; 256 MiB compressed input/member; 512 MiB total
expanded; compression ratio at most 100. Only stored/deflate unencrypted regular
files are accepted. Duplicate names, case aliases, directories, symlinks,
traversal, missing/extra members and hash/size mismatches fail before writes.
Identities have a 16 MiB limit and at most 500000 atom bindings. New instances
carry no fabricated atom mapping; verified reuse remains subject to the real
scientific loader's exhaustive evidence check.

`materialize_preparation` returns the checked frozen request, config and source
identity descriptors. The executor must validate real loader identity and
source box, execute science, independently verify the actual System, and return
terminal evidence. An absent executor returns executor_unavailable. No transport
fixture qualifies science. The wrapper accepts no arbitrary command, image,
OSS configuration or credentials.

The CLI connects the trusted classic executor. It checks the protein's existing
periodic source box before preparation, including when a ligand spans space;
missing, degenerate, unreduced or cutoff-incompatible boxes cannot be inferred.
Classic preparation exports v1 lineage and independently verifies the actual
System/topology before emitting `verification: passed`. Exported source atom and
residue UUIDs follow the qualified loader order: new instances use UUIDv5 under
the instance UUID; reuse passes exact content, loader, ordinal, element,
coordinate and residue-membership checks. Complete verified reuse attachments
are declared `identities/NNNN.json` artifacts for atomic acquisition by the API.

The ZIP retains both exact YAML byte streams and the immutable prepared bundle.
`resolved_parameters` separately records applied defaults, unit conventions,
actual force settings, versions, resource hashes, source/output box, translation,
hydrogen variants and QC gates. Default provenance distinguishes explicit input
from omitted defaults; an inactive hydrogen/solvent operation remains visible in
its applied switch and observed preparation steps.

QC counts all discoveries while displaying at most 50 per check. The complete
canonical UTF-8 result is bounded to 65,536 bytes; additional trailing display
issues may be omitted to fit that budget, with displayed counts and truncation
updated. Full archive QC, scientific verdicts, totals, thresholds, eligibility
and revision/location bindings remain unchanged. If the envelope without display
issues cannot fit, execution fails with a nonzero exit and no result.json,
retaining the archive as evidence of a technical transport failure. Successful
projection binds topology ordinals to verified prepared UUIDs and revision;
strict failure uses unlocated text with empty IDs and no prepared manifest.
Soft failure exports a verified prepared result with `eligible: false`.
Infrastructure exceptions emit controlled `wrapper_internal_error`; an absent
or killed wrapper still cannot fabricate science.


## Executor packaging

The owner-selected algorithm name is `md-prepare`; its Harbor repository is
`harbor.neobinder-internal.com/api/md-prepare`. The API operation and workflow
package ID remain `md.prepare`. Pin each registered executor to its verified
registry digest and exact source revision.

Argo overrides the resident image entrypoint with container command
`["neomd", "batch-prepare"]` and working directory `/work`. The fixed request,
source bundle and terminal outputs are the paths declared above; `/work` and
`/outputs` require writable isolated volumes. The current wheel supplies the CLI
and contracts. Reuse a verified runtime base through the source-only wheel
installation only when its environment-lock hash matches the unchanged
container lock; see [container packaging](../../../docker/README.md).

Initial CPU-only dev admission evidence qualifies only this batch operation.
It does not qualify resident CUDA execution; unavailable GPU validation cannot
satisfy the existing resident GPU build/qualification gate.

The frozen executor_binding transport accepts mutually exclusive published-image `{git_revision,image_digest}` and local-source `{git_revision,source_snapshot_sha256,binding_scope:"local_source"}` identities. Local source identity is checked by the resident host against its actual source/dependency snapshot and fixture qualification; it never claims container execution. This additive identity union changes no classic-source-box/v1 scientific parameter or result.
