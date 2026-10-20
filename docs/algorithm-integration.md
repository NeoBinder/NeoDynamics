# Algorithm integration and plugin boundaries

Status: accepted architecture direction. Existing SDK and service contracts
remain authoritative; a package manifest does not establish qualification,
remote preparation support or deployment.

## Responsibility

NeoDynamics owns molecular-dynamics science, system preparation, reusable
engine interfaces and qualified physical controls. `<NeoAPI-rs>` owns public
identity, durable orchestration and budget enforcement; `<NeoFlow>` owns fixed
pipelines. `<NeoHub>` and `<NeoCanova>` provide scientific views and Agent host
interaction through public contracts. MD is selected when justified by the
scientific question, not required for every candidate workflow.

## Two distinct extension mechanisms

| Mechanism | Trust and lifecycle |
| --- | --- |
| SDK plugin | Trusted code installed in the engine environment; registers methods, restraints, CVs or probes through `neomd.register()` or the `neomd` entry-point group |
| Algorithm package | Independently released executor image plus declared input/output, resource, provenance and qualification contracts |

Keep the existing SDK plugin surface. It is not an untrusted remote plugin
loader, and registration does not qualify a method for any resident service.
Public requests cannot load arbitrary Python, model code, probes or Plans.
See [architecture](architecture.md) for registry/driver responsibilities and
[Service and integration surface](integration-surface.md) for the boundary an
external distribution may rely on.

An external package may contain trusted SDK plugins. The qualified image fixes
which code is installed; changing that code requires a new executor release.
Updating a resident method still requires a resident deployment and affected
qualification. Independent algorithm releases do not imply live mutation of
running sessions or automatic compatibility with existing prepared systems.

## Package contract

The target package descriptor declares method/version, input and output
contract versions, parameters and units, resource bounds, execution mode,
artifact schemas, metric definitions, weight/data references and validation
evidence. This is a design boundary, not a new implemented manifest format.

Freeze input object versions, prepared identities, actual parameters including
defaults, seeds, image digest and scientific resources for a run. New scientific
behavior gets a new release even with compatible JSON. Retries preserve the
original binding; default rollback applies to new requests. Qualification,
publication and enablement are separate gates.

Scientific code and adapters stay in this repository or the owning algorithm
package. `<NeoAPI-rs>` validates access and contracts without importing engine
implementation. Compatible package iterations should eventually avoid control
plane releases; new object contracts or lifecycle modes still require
coordinated producer/consumer changes.

## Scientific objects and evidence

Preparation variants declare exact guarantees. An MD System preparation is
not interchangeable with a design-tool cleanup. Preserve source, prepared atom,
residue and particle identities and explicit correspondence across transforms.
Never infer identity from coordinates, array length or display numbering.
Selections bind to an exact object version; uncertain remapping blocks execution.

Results create immutable derived artifacts with inputs, settings and release
provenance. Metrics include method/version, units, reference state and conditions;
similarly named scores are not automatically comparable. A candidate can have
multiple structures and experiments; its identity is not the trajectory ID or
experimental sample ID.

[Prepared bundles](prepared-bundle.md) and [protein preparation](protein-preparation.md)
retain current contracts and qualification limits. In particular, bounded SDK
preparation does not imply
remote full preparation, and local result evidence does not imply OSS publication.
Golden and synthetic scale gates do not prove physical validity or scientific
throughput.

## Workflow and Agent selection

Use a fixed pipeline for a verified scientific protocol. Durable research
workflows with candidate fan-out, review, experiment feedback and budgets belong
to the accepted `<NeoAPI-rs>` orchestration direction. This does not expand
NeoDynamics' deferred multi-leg orchestration or excluded QM/MM scope.

Give each execution unit one retry/cancellation owner. The public control plane
coordinates a session or operation; NeoDynamics owns its admitted session,
engine stepping, checkpoint/resume and durable command semantics. Outer
workflow retries must not duplicate live sessions or replay uncertain mutations.

Client Agents may propose scientific parameters, inspect authorized results and
invoke public operations. They use the same identity, budget and scientific
validation as other callers. External capability discovery and HTTP Agent
entry documentation belong to `<NeoAPI-rs>`; internal service transport stays
private. Developer Agents select SDK extension, service change or external
package work using the boundary table above, then load only the affected
contract and qualification guide.

## Acceptance for independent delivery

Demonstrate compatible executor versions coexisting, a new run selecting a
qualified new release, an old run preserving its binding, and default rollback
without rewriting artifacts. Verify scientific correctness separately from
schema interoperability and deployment. Do not advertise pending methods,
preparation profiles or publication paths as available.
