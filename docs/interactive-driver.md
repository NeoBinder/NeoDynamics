# Interactive service driver

The service runtime lives in `neomd_server`; normal `neomd` CLI commands and
`neomd.md_run` remain available independently. The driver can be embedded in a
persistent service worker:

```python
from neomd_server.driver import open_session

# Construct and call this object only on its owning worker thread.
session = open_session(plan, platform="cpu")  # production default: CUDA
session.advance(100)
frame = session.frame()  # positions_nm, ordered particle_indices, time_ps
session.command({"type": "pause"})
session.command({"type": "resume"})
descriptor = session.checkpoint()
# Publish descriptor['artifacts'] together as one immutable recovery generation.
session.close()
```

`open_session` runs all RBFE ladder windows; `InteractiveSession` is the
single-window primitive. The host controls how frequently to advance and send
frames. It must keep one thread/process as the Context owner, and route commands
through that owner. Disconnecting a viewer does not implicitly pause or stop it.

Hosts publishing a command checkpoint use `session.apply_command(command)`.
It returns `CommandResult(status, checkpoint)`, retaining the exact descriptor
from a successful checkpoint or stop; other commands return `checkpoint=None`.
Publish it before advancing again. `command()` retains its status-only interface,
and an explicit `checkpoint()` still creates a new capture. Descriptors are never
cached across commands or inferred from a later observation.

The resident host can pass its `RunningBudget` as `running_budget` to
`open_session`. The service driver brackets running segments on the same owner
thread, suspends them for preparation/paused/finalization work and shares the
counter across RBFE windows. The host owns persistence and exhaustion dispatch;
this optional accounting hook does not alter scientific stepping or CLI physics.

Inspect `session.status()['controls']`, `parameters` and `command_limits` before
showing controls. Commands use explicit canonical units:

```python
session.command({
    "type": "set_forces", "indices": [17], "vectors": [[2.0, 0.0, 0.0]],
    "unit": "kJ/mol/nm", "ttl_steps": 1000, "ttl_ms": 500,
})
session.command({
    "type": "set_restraint", "name": "anchor", "indices": [17],
    "targets": [[1.0, 2.0, 3.0]], "k": 100.0,
    "position_unit": "nm", "k_unit": "kJ/mol/nm^2",
})
session.command({"type": "clear_restraints", "name": "anchor"})
```

Particle indices refer to the prepared System, not display atom IDs. A drag
force expires automatically by either deadline. Named positional restraints
remain until explicitly cleared and survive restart; transient drag forces do
not. SMD target edits require an advertised parameter and explicit
`replace_schedule=True`. Their acknowledgement contains the step and control
revision at which they took effect.

Recovery uses a plan with `continue_md=True` and the complete saved artifact
generation, on the same supported scientific platform/system/version. Copy a
generation into a new mutable working directory before resuming it. Hash or
state mismatches raise an error before scientific tapes are changed. The host
owns durable generation publication, session ownership, user access and transport.
See [ADR-0008](adr/0008-interactive-service-driver.md) for method qualifications,
minimization semantics and the scientific/service boundary.
