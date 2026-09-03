"""neomd-gamd-drill — the GAMD plugin drill (v2 migration plan §5 item 2.9).

GAMD itself is a Non-Goal (plan §2): this package exists only to validate
the v2 extension rack from OUTSIDE the core.  It mirrors the shape of
the in-tree method triple ``neomd.methods.metadynamics`` — a
``Method(schema, prepare)`` whose ``prepare(kernel=..., plan=...,
sink=..., logger=...) -> PreparedMethod`` is what ``driver.drive()``
dispatches through ``registry.get("method", name).prepare(...)`` — while
carrying placeholder physics only:

* ONE bias is installed through ``kernel.install_bias``: a
  ``CustomCentroidBondForce`` whose energy expression is the constant
  ``"0.0*k_drill"`` (zero by construction — no GAMD boost potential, just
  proof that a plugin can hand a BiasIR to any kernel);
* the loop is run by the DRIVER (``driver.run_prepared_method``) with the
  drill's ``on_step`` hook counting a boost "update" every ``frequency``
  steps (the Wave-2 method seam metadynamics uses for hill deposition);
* a drill artifact ``gamd_drill.log`` is appended through the run ``sink``
  after every update;
* the ``finish`` half of the contract returns a value mirroring the
  metadynamics ``MethodResult`` attribute contract
  (``steps_done`` / ``fgroup`` / ``positions_sha256`` plus the
  drill-specific ``n_updates``).

Import side effect (the plugin contract): importing this module registers
``GAMD_METHOD`` under ``register("method", "gamd", ...)`` AND its plan
section under ``register("plugin", "gamd_drill", PluginSection(...))`` —
the plugin plan-schema namespace (ADR-0002): a plan carries the drill's
settings as ``plugins.gamd_drill.{boost_factor, frequency, k_drill}``,
plan.py validates the section's name and keys against this declaration
(collect-all, with did-you-mean), and the section rides ``plan.raw`` into
the fingerprint.  The registry makes re-imports idempotent and rejects
collisions with a different object, so no guard is needed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from neomd.kernel.port import BiasIR, Param
from neomd.registry import PluginSection, register

__all__ = [
    "Method",
    "GAMDResult",
    "GAMD_METHOD",
    "PLUGIN_SECTION",
    "LABEL",
    "NAMESPACE",
    "LOG_FILENAME",
    "SCHEMA",
    "DEFAULT_SETTINGS",
]

LABEL = "gamd"
NAMESPACE = "gamd_drill"  # the plugins.<NAMESPACE>.* plan section (ADR-0002)
LOG_FILENAME = "gamd_drill.log"

LOG = logging.getLogger("neomd_gamd_drill")

#: placeholder settings (no physics): boost_factor is recorded in the log and
#: never used, k_drill multiplies a hard-coded 0.0, frequency drives the
#: on_step cadence of the counted "updates"
DEFAULT_SETTINGS = {
    "boost_factor": 1.0,  # dimensionless (a real GAMD: sigma0 / dV)
    "frequency": 10,  # steps between counted boost "updates"
    "k_drill": 0.0,  # amplitude of the constant-energy placeholder
}


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method").

    The drill's own mirror of ``neomd.methods.metadynamics.Method`` — the
    registry stores whatever object it is handed and the driver only touches
    ``entry.prepare(...)``, so a plugin does NOT need the in-tree dataclass.
    The attribute contract is identical:

    ``prepare(kernel=..., plan=..., sink=..., logger=...) -> PreparedMethod``
    """

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class GAMDResult:
    """Outcome of one drill run (drive() appends it to RunOutcome.results).

    Mirrors the metadynamics ``MethodResult`` attribute contract; the drill's
    count of interest is ``n_updates`` instead of ``n_hills``.
    """

    steps_done: int  # final absolute step count
    fgroup: int  # force-group id the kernel assigned the installed bias
    n_updates: int  # boost "updates" counted on the on_step seam
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


def _settings(plan) -> dict:
    """Resolve the drill's settings from the ``plugins.gamd_drill`` section.

    ADR-0002: the namespace is first-class — plan validation already proved
    the section's name and keys against :data:`PLUGIN_SECTION` by the time a
    Plan reaches ``prepare``, so this only merges the section over
    :data:`DEFAULT_SETTINGS`.  Values (types, ranges) stay the plugin's own
    business, exactly like a method triple reading its plan keys.
    """
    settings = dict(DEFAULT_SETTINGS)
    section = (getattr(plan, "plugins", None) or {}).get(NAMESPACE)
    if section:
        settings.update(dict(section))
    return settings


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'gamd'.

    Returns a :class:`~neomd.driver.PreparedMethod`: the drill's boost
    logger rides the ``on_step`` seam and the DRIVER runs the loop with the
    reporting it owns (the drill pins no registry tape — its log is written
    here, outside the probe list).
    """
    from neomd.driver import CHECKPOINT_FILENAME, PreparedMethod

    log = LOG if logger is None else logger
    settings = _settings(plan)

    frequency = int(settings["frequency"])
    if frequency < 1:
        raise ValueError(
            f"gamd drill frequency must be >= 1, got {frequency}")

    # -- one placeholder bias: constant zero energy on atoms 0-1 ------------
    # (the v1 GAMD essential-energy rewrite would live in this expression;
    #  the drill pins it to 0.0*k_drill so both kernels compile it but no
    #  physics happens)
    bias = BiasIR(
        kind="CustomCentroidBondForce",
        energy="0.0*k_drill",
        params={"k_drill": Param(float(settings["k_drill"]), "dimensionless")},
        groups=[[0], [1]],
        periodic=False,
        label=LABEL,
    )
    fgroup = kernel.install_bias(bias)

    # -- the drill artifact, appended per boost update like a reporter ------
    if sink is not None:
        with sink.text_writer(LOG_FILENAME) as handle:
            handle.write(
                "# neomd GAMD plugin drill (v2 plan §5 item 2.9)\n"
                f"# boost_factor={settings['boost_factor']} "
                f"frequency={frequency} fgroup={fgroup}\n")

    updates: list[int] = []  # steps at which a boost "update" fired

    def on_step(step: int, view) -> None:
        updates.append(int(step))
        log.info("gamd drill boost update %d at step %d", len(updates), step)
        if sink is not None:
            with sink.text_writer(LOG_FILENAME) as handle:
                handle.write(f"{int(step)}\t{len(updates)}\n")

    def finish(result):
        # v1 save_last: checkpoint at run end (mirrors the metadynamics triple)
        if sink is not None:
            sink.write_bytes(CHECKPOINT_FILENAME, kernel.snapshot())
        return GAMDResult(
            steps_done=result.steps_done,
            fgroup=fgroup,
            n_updates=len(updates),
            positions_sha256=result.positions_sha256,
        )

    return PreparedMethod(
        on_step=on_step,
        on_step_interval=frequency,
        fgroups={LABEL: [fgroup]},
        finish=finish,
    )


SCHEMA = {
    "required": {
        "steps": "int, total steps (plan-level key)",
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        # The plugin plan-schema namespace (ADR-0002): the drill's settings
        # ride under plugins.gamd_drill.*, validated against PLUGIN_SECTION.
        "plugins.gamd_drill": (
            "mapping with boost_factor (dimensionless placeholder), "
            "frequency (steps between counted boost updates, default 10), "
            "k_drill (constant-energy amplitude, default 0.0)"),
        "output.*": ("output_dir + state/trajectory/checkpoint intervals "
                     "(plan-level; gamd_drill.log always appends one row "
                     "per boost update)"),
    },
}

#: the method knowledge triple this distribution contributes to the rack
GAMD_METHOD = Method(schema=SCHEMA, prepare=_prepare)

#: the plan section it owns under plugins.<NAMESPACE>.* (ADR-0002) —
#: registered next to the method triple; mirrors the SCHEMA shape
PLUGIN_SECTION = PluginSection(
    required={},
    optional={
        "boost_factor": "dimensionless placeholder (a real GAMD: sigma0/dV)",
        "frequency": "steps between counted boost updates (default 10)",
        "k_drill": "constant-energy amplitude of the placeholder bias (0.0)",
    },
)

# -- the plugin contract: importing this module self-registers ----------------
register("method", LABEL, GAMD_METHOD)
register("plugin", NAMESPACE, PLUGIN_SECTION)
