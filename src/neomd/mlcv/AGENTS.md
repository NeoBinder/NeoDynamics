# ML collective variables

Featurization reuses public CV evaluation. Pool TICA runs without crossing run
boundaries. Keep numpy feature/training paths torch-free; gated TorchScript
export must reproduce `apply_model` closely. Simulation-core injection is a
separate design decision in project-root `docs/adr/` ADR-0006, not implicit
scope for changes to these tools.
