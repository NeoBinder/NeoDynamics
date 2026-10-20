# Extending NeoDynamics

Use plugins to add methods and validated settings. Embedded hosts can also
customize reporting without replacing the scientific stepping loop.

## Custom reporting

Trusted in-process hosts can pass `probes` to
`neomd.driver.prepare_method_stepper`. Omitting it or passing `None` keeps
the existing default, restraint and enabled method reporting probes; an explicit
sequence replaces that reporting list. Scientific hooks, method state,
checkpoints and finalization remain independent, so `probes=[]` does not mean
file-free execution. This engine interface does not change CLI options or output
defaults.

An out-of-tree host may supply its own explicit reporting assembly this way
while still writing the configured archive files. File reports and any live
observation channel remain separate responsibilities of that host; the engine
guarantees the reporting seam described above.

## Method plugins

Register a knowledge triple — the registry is the public plugin surface
(`neomd.register`). A method plugin mirrors the metadynamics shape
(`schema` + `prepare(kernel, plan, sink, logger) -> PreparedMethod`):
install your biases, then hand the driver your `on_step` hook, tape probes
and a `finish` — the driver runs the loop and owns all reporting (the
restraint tape, and whether your tapes run at all). Importing your
module self-registers. This minimal example installs a zero-energy placeholder
on particles 0 and 1; replace it with your qualified scientific implementation:

```python
from types import SimpleNamespace
from neomd.driver import PreparedMethod
from neomd.kernel.port import BiasIR, Param
from neomd.registry import PluginSection, register

SCHEMA = {
    "required": {"steps": "int", "temperature": "number (K)"},
    "optional": {"plugins.my_method": "method settings"},
}

def prepare(kernel, plan, sink=None, logger=None):
    settings = (getattr(plan, "plugins", None) or {}).get("my_method", {})
    k = float(settings.get("k", 0.0))
    bias = BiasIR(
        kind="CustomCentroidBondForce",
        energy="0.0*k",                       # your potential expression here
        params={"k": Param(k, "dimensionless")},
        groups=[[0], [1]],
        periodic=False,
        label="my_method",
    )
    fgroup = kernel.install_bias(bias)         # opaque force-group id

    def finish(result):                        # end-of-run artifacts
        from neomd.driver import CHECKPOINT_FILENAME
        if sink is not None:
            sink.write_bytes(CHECKPOINT_FILENAME, kernel.snapshot())
        return result

    return PreparedMethod(
        fgroups={"my_method": [fgroup]},       # informational
        finish=finish,
        # on_step=..., on_step_interval=..., tapes={...} as needed
    )

register("method", "my_method", SimpleNamespace(schema=SCHEMA, prepare=prepare))
register("plugin", "my_method", PluginSection(
    required={}, optional={"k": "dimensionless placeholder amplitude"},
))
```

Settings live under `plugins.my_method`, with accepted keys declared through
`PluginSection`. Plan validation checks names and keys; `prepare` owns value
types and ranges. Required-key checks also run with `neomd validate --check-files`.
The settings participate in the plan fingerprint. Load the plugin before
constructing a `Plan` directly; the facade scans installed entry points for you.
See [ADR 0002](adr/0002-plugin-plan-schema-namespace.md) for the full contract.

For installable distributions, declare the `"neomd"` entry-point group:

```toml
[project.entry-points."neomd"]
my_method = "my_package"      # importing my_package self-registers
```

A complete, tested drill lives in
[plugin drill](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/gamd_drill) — an out-of-tree mini-distribution
validating registration, entry-point discovery and `drive()` dispatch on both
the fake and openmm kernels.

See [architecture](architecture.md) for kernel boundaries and [configuration](reference/configuration.md) for supported plan fields.
