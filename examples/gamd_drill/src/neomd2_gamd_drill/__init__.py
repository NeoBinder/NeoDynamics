"""neomd2-gamd-drill — the GAMD plugin drill (v2 migration plan §5 item 2.9).

GAMD itself is a Non-Goal (plan §2): this package exists only to validate
the v2 extension rack from OUTSIDE the core.  It mirrors the shape of the
in-tree method triple ``neomd2.methods.metadynamics`` — a ``Method(schema,
run)`` whose ``run(kernel=..., plan=..., sink=..., logger=...) -> result``
is what ``driver.drive()`` dispatches through
``registry.get("method", name).run(...)`` — while carrying placeholder
physics only:

* ONE bias is installed through ``kernel.install_bias``: a
  ``CustomCentroidBondForce`` whose energy expression is the constant
  ``"0.0*k_drill"`` (zero by construction — no GAMD boost potential, just
  proof that a plugin can hand a BiasIR to any kernel);
* the loop runs through ``driver.run_md`` with an ``on_step`` hook that
  counts a boost "update" every ``frequency`` steps (the Wave-2 method seam
  metadynamics uses for hill deposition);
* a drill artifact ``gamd_drill.log`` is appended through the run ``sink``
  after every update;
* the return value mirrors the metadynamics ``MethodResult`` attribute
  contract (``steps_done`` / ``fgroup`` / ``positions_sha256`` plus the
  drill-specific ``n_updates``).

Import side effect (the plugin contract): importing this module registers
``GAMD_METHOD`` under ``register("method", "gamd", ...)``.  The registry
makes re-imports idempotent and rejects collisions with a different object,
so no guard is needed here.

The ``gamd_set`` schema question (outcome, see README):
``plan.KNOWN_KEYS`` is a CLOSED whitelist — a top-level ``gamd_set`` section
is rejected with ``ConfigKeyError`` before any method sees the plan, and v2
has no generic plugin namespace yet.  ``_settings`` therefore resolves the
drill's settings (a) from ``plan.raw["gamd_set"]`` *tolerantly* — the natural
future spelling, picked up automatically the day the whitelist opens —
falling back to (b) ``plan.meta_set["gamd_drill"]`` (riding inside an
existing whitelisted mapping section, the documented v2 extension path) and
finally to (c) defaults.  The core package is never edited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from neomd2.kernel.port import BiasIR, Param
from neomd2.registry import register

__all__ = [
    "Method",
    "GAMDResult",
    "GAMD_METHOD",
    "LABEL",
    "LOG_FILENAME",
    "SCHEMA",
    "DEFAULT_SETTINGS",
]

LABEL = "gamd"
LOG_FILENAME = "gamd_drill.log"

LOG = logging.getLogger("neomd2_gamd_drill")

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
    """One method knowledge triple: schema + run (registry kind "method").

    The drill's own mirror of ``neomd2.methods.metadynamics.Method`` — the
    registry stores whatever object it is handed and the driver only touches
    ``entry.run(...)``, so a plugin does NOT need the in-tree dataclass.  The
    attribute contract is identical:

    ``run(kernel=..., plan=..., sink=..., logger=...) -> result``
    """

    schema: dict
    run: Callable


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
    """Resolve the drill's settings from ``plan``, tolerantly.

    Precedence (see module docstring — the closed-whitelist outcome):

    1. ``plan.raw["gamd_set"]`` — the natural spelling; today this is empty
       because ``Plan.from_dict`` rejects the key outright (ConfigKeyError),
       but the getter tolerates it so the plugin picks the section up the
       day the core whitelist opens (no plugin change needed);
    2. ``plan.meta_set["gamd_drill"]`` — the documented v2 ride-along path:
       ``meta_set`` is a whitelisted mapping section and plan validation
       checks its TYPE, not its keys, so a third-party method may carry its
       own sub-section there;
    3. ``DEFAULT_SETTINGS``.
    """
    settings = dict(DEFAULT_SETTINGS)
    carrier = (getattr(plan, "raw", None) or {}).get("gamd_set")
    if not carrier:
        carrier = dict(getattr(plan, "meta_set", None) or {}).get("gamd_drill")
    if carrier:
        settings.update(dict(carrier))
    return settings


def _run(kernel, plan, sink=None, logger=None) -> GAMDResult:
    """Registry entry point — drive() calls this for method 'gamd'."""
    from neomd2.driver import CHECKPOINT_FILENAME, _default_probes, run_md

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
                "# neomd2 GAMD plugin drill (v2 plan §5 item 2.9)\n"
                f"# boost_factor={settings['boost_factor']} "
                f"frequency={frequency} fgroup={fgroup}\n")

    updates: list[int] = []  # steps at which a boost "update" fired

    def on_step(step: int, view) -> None:
        updates.append(int(step))
        log.info("gamd drill boost update %d at step %d", len(updates), step)
        if sink is not None:
            with sink.text_writer(LOG_FILENAME) as handle:
                handle.write(f"{int(step)}\t{len(updates)}\n")

    result = run_md(kernel, plan, _default_probes(plan, sink),
                    on_step=on_step,
                    on_step_interval=frequency,
                    logger=log)

    # v1 save_last: checkpoint at run end (mirrors the metadynamics triple)
    if sink is not None:
        sink.write_bytes(CHECKPOINT_FILENAME, kernel.snapshot())
    return GAMDResult(
        steps_done=result.steps_done,
        fgroup=fgroup,
        n_updates=len(updates),
        positions_sha256=result.positions_sha256,
    )


SCHEMA = {
    "required": {
        "steps": "int, total steps (plan-level key)",
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        # The natural spelling.  NOTE: plan.KNOWN_KEYS is a closed whitelist
        # today — a top-level gamd_set key raises ConfigKeyError at Plan
        # construction; see README "The gamd_set schema question".
        "gamd_set": (
            "mapping with boost_factor (dimensionless placeholder), "
            "frequency (steps between counted boost updates, default 10), "
            "k_drill (constant-energy amplitude, default 0.0)"),
        # The documented v2 ride-along carrier while the whitelist is closed.
        "meta_set.gamd_drill": ("same keys as gamd_set, carried inside the "
                                "whitelisted meta_set mapping section"),
        "output.*": ("output_dir + state/trajectory/checkpoint intervals "
                     "(plan-level; gamd_drill.log always appends one row "
                     "per boost update)"),
    },
}

#: the knowledge triple this distribution contributes to the rack
GAMD_METHOD = Method(schema=SCHEMA, run=_run)

# -- the plugin contract: importing this module self-registers ----------------
register("method", LABEL, GAMD_METHOD)
