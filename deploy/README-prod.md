# First production NeoDynamics deployment

`app-prod.yaml`, `migrate-prod.yaml` and `runtime-config-prod.yaml` are for
namespace `api`. The app and migration manifests must both pin the **same**
Harbor sha256 that ran on `api-dev`. The current pin is
`sha256:885da45a933f31a408bfd7dfd18f3348f412ed313f566311fb195e170cac21c1`.
Verify the final R01 result and every running pod imageID before applying;
if the qualified image changes, update both pins. The
production provider identity is
`194a5d82-cc13-4423-9754-c1245a79b751` and must match the NeoAPI API and
worker MD settings.

## Prerequisites

- Provision a separate `neodynamics` production PostgreSQL database and
  `neodynamics_migrator`/`neodynamics_runtime` roles with the schema ownership
  and runtime grants in [the deployment guide](README.md#database-and-secret-interfaces).
  Do not point either role at NeoAPI's database or copy staging credentials.
- Provision `neodynamics-runtime-secret` with `NEOMD_SERVER_PG_DSN` and
  `service-token`, and `neodynamics-migration-secret` with its distinct
  `NEOMD_SERVER_PG_DSN`. The service token must match the production-only
  `NEOAPI_MD_SERVICE_TOKEN` in both NeoAPI API and worker Secrets.
- Provision a distinct `md-publication-auth` Secret with a `token` key. Its
  value must match the production NeoAPI worker reverse-publication Secret.
  Neither Secret value belongs in Git, logs or command arguments.
- Confirm `nvidia` RuntimeClass, one available GPU slice on `kube-gpu`,
  `longhorn` StorageClass and the `harbor-robot-pull` Secret. The new PVC is
  300 GiB, with a 240 GiB ledger cap; `Recreate` keeps one writer.
- Production NeoAPI manifests must include the private
  `neoapi-md-publication.api.svc.cluster.local:8083` Service and the ingress
  NetworkPolicy rule for pods labeled `app=neodynamics,component=resident-md`.
  The production provider UUID and origin in the ConfigMap must match them.

## Activation

1. Review the exact manifest diff, image source inventory and all running
   api-dev pod imageIDs. Verify both pinned Dynamics images against the final
   R01 digest. Run client/server dry-runs on the three files and confirm Secrets
   exist by name and expected key only, without displaying values.
2. Apply `runtime-config-prod.yaml`; run `migrate-prod.yaml` once while no
   resident service is running. Wait for Job completion, inspect redacted
   migration logs, verify schema version and runtime/default grants. A repeat
   migration requires a deliberate new Job name and migration review.
3. Apply `app-prod.yaml` and wait for the one ready Deployment replica.
   Compare the actual running `imageID` with the qualified registry digest.
   Verify authenticated private capability response, including intended CUDA
   methods. Complete the NeoAPI schema migration gate before rolling its API
   and worker with their MD transport settings.
4. Verify both NeoAPI API replicas, the worker and NeoDynamics at their
   qualified digests. Through the production public API, run one bounded,
   owner-scoped preparation/result-publication smoke. Verify result hashes and
   worker publication reachability, and record any failure before retrying.

## Rollback boundary

The existing production API and worker image is
`harbor.neobinder-internal.com/api/neoapi-rs@sha256:c634ec76743c7c738d073c6e7b9a23bf7a9892692990c381bd7f18457c668342`.
Before applying NeoAPI migrations 25–32, retain the reviewed production
database backup/schema snapshot and verify whether this old binary remains
compatible with the migrated schema. A disposable v24→v32 rehearsal showed
its API health and worker schema check pass at v32, but production readiness
and active executor behavior still need release review. The migrations add workflow cleanup
triggers and MD tables; a Deployment image rollback does **not** undo database
effects. Do not run a speculative down migration or erase new MD records.

Keep API and worker at the old digest while the first Dynamics database and
service are brought up. If Dynamics fails before the API/worker cutover, leave
the old public workloads running, stop the new service only after confirming
no active sessions/publication effects, and preserve its DB/PVC for diagnosis.
If a later cutover fails, first inspect and settle any active MD sessions and
publication work; return API/worker to the old digest only when the reviewed
schema-compatibility gate permits it. Otherwise hold traffic and use the
approved database recovery procedure, preserving the exact state and logs.

The staging operational limits are copied in the production ConfigMap with a
separate pool identity. A bounded 24-hour Min/Eq finalized-state retention
window remains capacity-reserved so phase continuation can validate exact
sealed bytes; the guarded cleanup releases that reservation after expiry.
Admission closes before the source deadline by the configured queue TTL,
startup timeout and a 60-second margin (18 minutes with the current settings).
Exceptional delays after admission can still fail closed if the source is
swept before execution. Start dependent phases well inside the 24-hour window
and treat a terminal source-unavailable result as a fresh-run decision, never
as authority to reuse unverified bytes.
