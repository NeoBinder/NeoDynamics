"""Gaussian-accelerated MD (GaMD) — a method knowledge triple (schema +
prepare + the staged calibration/boost run).  See ADR-0005
(docs/adr/0005-gamd-boost-seam.md), ADR-0010 (per-stage steps input and
boundary restart), ADR-0011 (each stage is an independent LEG with its own
Context and step axis) and docs/methods/gamd.md.  Registers method: ``gamd``.
Never imports openmm (methods stay kernel-agnostic; the port's kJ/mol
unit conventions throughout).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from neomd.kernel.port import (
    BoostChannelIR,
    BoostOps,
    PartitionOps,
    provides,
)
from neomd.registry import register

__all__ = [
    "Method",
    "MethodResult",
    "GamdRun",
    "LABEL",
    "TAPE_FILENAME",
    "CALIBRATION_FILENAME",
    "DEFAULT_SETTINGS",
    "DEFAULT_SIGMA0_KCAL_MOL",
    "STAGE_NAMES",
    "read_gamd_trace",
    "reweight_observable",
]

LABEL = "gamd"
TAPE_FILENAME = "gamd.tsv"
CALIBRATION_FILENAME = "gamd_calibration.json"

LOG = logging.getLogger("neomd.methods.gamd")

#: the per-stage ``steps`` vocabulary (ADR-0010): each value is that stage's
#: own step requirement AND its leg's length, never an absolute count (ADR-0011
#: restarts the kernel clock at every stage).  The first four are the
#: kernel-driven preparation legs; ``production`` is the driver's leg.
STAGE_NAMES = ("unbiased_eq", "unbiased_production", "biased_eq", "biased_eq2")
ALL_STAGE_NAMES = STAGE_NAMES + ("production",)

#: Statistics stages.  ``unbiased_production`` samples the un-boosted
#: ensemble; the boosted equilibrations sample the original channel energy
#: under the parameter set fixed at their start.  Parameters are handed over
#: only at a stage boundary (never inside a stage): the source stage's
#: samples calibrate the NEXT boost stage (ADR-0010).
_SAMPLE_PHASES = ("unbiased_production", "biased_eq", "biased_eq2")

#: Checkpoint-metadata key holding the leg the snapshot belongs to.  A leg's
#: clock is relative to its own Context and its RNG stream is that Context's,
#: so the index is state and must travel with the blob (ADR-0011): the journal
#: alone cannot disambiguate a clock of 0 between two adjacent leg entries.
_LEG_METADATA_KEY = "gamd_leg"

#: Stage whose samples calibrate the next boost stage.  ``biased_eq`` hands
#: over only when ``biased_eq2`` has steps; ``biased_eq2`` never recalibrates,
#: so production keeps the parameter set its last boost stage started with.
_HANDOFF_SOURCE = {"unbiased_production": "biased_eq", "biased_eq": "biased_eq2"}

#: Steps between cumulative-journal flushes inside one preparation
#: call.  The driver now stops at every reporting-probe boundary, so
#: flushing per stop would rewrite an ever-growing journal O(n) times;
#: the driver's own checkpoint cadence still flushes explicitly.
_FLUSH_INTERVAL = 5000

#: The built-in boost-strength default: 6 kcal/mol (gamd-openmm's literature
#: value for the total channel).  It applies when ``gamd.sigma0`` is absent; a
#: number written in the YAML is interpreted in ``gamd.energy_unit`` (kJ/mol
#: unless that key says otherwise).  Everything internal is kJ/mol.
DEFAULT_SIGMA0_KCAL_MOL = 6.0

#: the ``gamd:`` plan section defaults (per-stage lengths live in the
#: plan-level ``steps`` mapping, see STAGE_NAMES)
DEFAULT_SETTINGS = {
    "mode": "total",  # "total" | "dual"  (explicit channels: see below)
    # None = the literature default, ``DEFAULT_SIGMA0_KCAL_MOL`` kcal/mol; a
    # value written in the YAML is a plain number in ``energy_unit`` (kJ/mol
    # unless that key says otherwise)
    "sigma0": None,
    "channels": None,  # optional [{label, groups}] explicit channel defs
    "boosts": None,  # ordered atom/type partitions; never implicit remainder
    "threshold_policy": "upper",  # preserves historical upper-with-lower-fallback
    "energy_unit": "kJ/mol",
    "exchange": None,
    "calibration_input": None,  # external energy tape / gamd_calibration.json
    "restart_from": None,  # continue_md: rerun from this stage's boundary
}

#: The sampling/recording cadences are OUTPUT detail, so they live in the
#: method's own output namespace (``output.gamd``, ADR-0011) rather than in
#: the science section or in the analysis.  The defaults are deliberately
#: coarse: the CE2 analysis can only use rows a saved frame matches, so a
#: denser boost tape adds bytes, not information.  ``neomd.plan.gamd_output``
#: owns the defaults; this mirrors them for direct GamdRun construction.
OUTPUT_DEFAULTS = {"report": True, "sample_interval": 1000,
                   "trace_interval": 5000}

_VALID_MODES = ("total", "dual")

#: cadence keys that moved into the method's output namespace (ADR-0011):
#: the plan validator reports these with did-you-mean; kept here so direct
#: GamdRun construction fails with the same guidance
_MOVED_OUTPUT_KEYS = {
    "calibration_interval": "output.gamd.sample_interval",
    "adaptation_interval": "output.gamd.sample_interval",
    "frequency": "output.gamd.trace_interval",
}

#: legacy pre-ADR-0010 stage keys -> their per-stage steps spelling (the
#: plan validator reports these with did-you-mean; kept here so direct
#: GamdRun construction fails with the same guidance)
_MOVED_STAGE_KEYS = {
    "preparation_steps": "unbiased_eq",
    "calibration_steps": "unbiased_production",
    "boost_preparation_steps": "biased_eq",
    "equilibration_steps": "biased_eq2",
}


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method")."""

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one GaMD run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # steps run in the PRODUCTION leg (legs restart at 0, ADR-0011)
    channels: dict  # label -> {threshold, k} actually pushed to the kernel
    mean_boost: dict  # label -> mean ΔV (kJ/mol) over the production rows
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# the calibration math — pure functions (one definition point, test-parseable)
# ---------------------------------------------------------------------------


def _output_settings(plan) -> dict:
    """``output.gamd`` with defaults applied (one reader; plan owns them)."""
    from neomd.plan import gamd_output

    section = gamd_output(plan)
    return {key: section.get(key, default)
            for key, default in OUTPUT_DEFAULTS.items()}


def _channel_params(samples: np.ndarray, sigma0: float, threshold_policy="upper") -> dict:
    """Literature (E, k) selection from calibration samples (kJ/mol).

    Lower bound:  ``E = Vmax``, ``k0 = min(1, (σ0/σV)·(Vmax−Vmin)/(Vmax−Vavg))``.
    Upper bound:  ``k0 = (1−σ0/σV)·(Vmax−Vmin)/(Vavg−Vmin)``,
    ``E = Vmin + (Vmax−Vmin)/k0`` — used when its k0 lands in (0, 1],
    else the lower bound.  The effective harmonic constant is
    ``k = k0/(Vmax−Vmin) ∈ (0, 1/(Vmax−Vmin)]``, which bounds the kernel's
    force scaling ``s = 1 − k(E−P)`` to [0, 1] (forces never flip).

    Returns ``{"threshold": E, "k": k, "vmax": ..., "vmin": ...,
    "vavg": ..., "sigma": ...}``; a degenerate sample range (σV = 0, the
    system's energy does not vary — e.g. a frozen fixture) keeps the
    channel at ZERO strength rather than dividing by zero.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("GaMD calibration samples must be nonempty and finite")
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("GaMD sigma0 must be finite and positive")
    if threshold_policy not in ("lower", "upper"):
        raise ValueError("GaMD threshold_policy must be lower or upper")
    vmax = float(np.max(samples))
    vmin = float(np.min(samples))
    vavg = float(np.mean(samples))
    sigma = float(np.std(samples))
    return _params_from_statistics(vmin, vmax, vavg, sigma, sigma0, threshold_policy)


def _params_from_statistics(vmin, vmax, vavg, sigma, sigma0, threshold_policy):
    """Parameter selection shared by batch samples and online Welford moments."""
    result = {"threshold": 1e99, "k": 0.0, "vmax": vmax, "vmin": vmin,
              "vavg": vavg, "sigma": sigma,
              "bound": None if sigma == 0.0 else "lower",
              "threshold_policy": threshold_policy, "fallback": False,
              "disabled_reason": None}
    if sigma == 0.0 or vmax <= vmin or not (vmax > vavg > vmin):
        result["disabled_reason"] = "degenerate_statistics"
        return result  # no usable spread: stay at zero strength
    k0_lower = min(1.0, (sigma0 / sigma) * (vmax - vmin) / (vmax - vavg))
    k0_upper = (1.0 - sigma0 / sigma) * (vmax - vmin) / (vavg - vmin)
    if threshold_policy == "upper" and 0.0 < k0_upper <= 1.0:
        # upper bound (threshold above the sample mean): E from k0
        threshold = vmin + (vmax - vmin) / k0_upper
        k0 = k0_upper
        result["bound"] = "upper"
    else:
        # lower bound (threshold at Vmax): gamd-openmm's E = Vmax form
        threshold = vmax
        k0 = k0_lower
        result["bound"] = "lower"
        result["fallback"] = threshold_policy == "upper"
    if not 0.0 < k0 <= 1.0:  # pragma: no cover - k0_lower min() guards this
        return result
    result["threshold"] = float(threshold)
    result["k"] = float(k0 / (vmax - vmin))
    return result


def _tape_energy_samples(path, labels):
    """Extract per-channel energy samples (kJ/mol) from an external tape.

    Accepts a ``gamd.tsv`` boost trace (``<label>__energy`` columns) or a
    plain ``state.tsv`` energy log (its ``potential`` column feeds ONLY the
    ``total`` channel — per-channel energies need a gamd tape).  The file
    is user-specified provenance: which run's ensemble these samples
    describe is the user's responsibility, recorded via the sha256 the
    caller stores alongside the selected parameters.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"calibration_input {path!r} is empty")
    header = None
    for line in lines:
        if line.startswith("#"):
            header = line.lstrip("#").split()
    if header is None:
        raise ValueError(f"calibration_input {path!r} has no '#' header column line")
    rows = [line for line in lines if not line.startswith("#")]
    if not rows:
        raise ValueError(f"calibration_input {path!r} has no data rows")
    samples = {}
    for label in labels:
        column = f"{label}__energy"
        if column in header:
            index = header.index(column)
        elif label == "total" and "potential" in header:
            index = header.index("potential")
        else:
            raise ValueError(
                f"calibration_input {path!r} lacks a {column!r} column for "
                f"boost channel {label!r} (a plain state.tsv can only feed "
                "the total channel)")
        values = np.asarray([float(row.split("\t")[index]) for row in rows],
                            dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"calibration_input {path!r} has nonfinite energies")
        samples[label] = values
    return samples


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class GamdRun:
    """One GaMD execution over a kernel (install → staged calibrate → boost).

    The five per-stage lengths come from the plan-level ``steps`` mapping
    (``plan.gamd_steps``; ADR-0010): each value is that stage's own step
    requirement AND the length of that stage's leg, because ADR-0011 gives
    every stage its own Context and its own step axis starting at 0 (the
    driver's production target is therefore the production leg's length, see
    :attr:`GamdStages.total_steps`).  Continue semantics (source
    classification lives in ``prepare()``):

    * fresh — no ``continue_md``: all five legs run from step 0.
    * adopt — ``continue_md`` from a non-gamd checkpoint (external eq run):
      coordinates/velocities are inherited, but the external step axis is
      DROPPED (``start_leg(0)`` zeroes the clock) because legs have no global
      axis; all five legs run from 0.
    * resume — ``continue_md`` into a gamd output directory: per-leg
      completed accounting, only the unfinished legs run; stage lengths may
      differ from the original run (extension by growing ``production``).
    * restart — ``continue_md`` + ``gamd.restart_from: <stage>``: re-enter
      that stage's leg from its boundary checkpoint; parameters are pinned
      to the source run's calibration at that boundary (or recomputed /
      adopted from ``gamd.calibration_input`` when given).

    Construct directly for artifact access (``channels`` /
    ``read_gamd_trace``); ``prepare()`` is the registry entry drive()
    dispatches; ``run()`` is the direct-construction convenience (prepare
    + the driver's method loop) returning the :class:`MethodResult`.
    """

    def __init__(self, kernel, plan, sink=None, logger=None):
        self.kernel = kernel
        self.plan = plan
        self.sink = sink
        self.log = LOG if logger is None else logger

        settings = dict(DEFAULT_SETTINGS)
        section = dict(getattr(plan, "gamd", None) or {})
        moved = sorted(set(section) & set(_MOVED_STAGE_KEYS))
        if moved:
            raise ValueError(
                f"gamd section has pre-ADR-0010 stage keys {moved}: they moved "
                "to the per-stage steps mapping "
                f"({', '.join(f'steps.{_MOVED_STAGE_KEYS[k]}' for k in moved)})")
        relocated = sorted(set(section) & set(_MOVED_OUTPUT_KEYS))
        if relocated:
            raise ValueError(
                "gamd section has output-cadence keys "
                f"{ {key: _MOVED_OUTPUT_KEYS[key] for key in relocated} }: "
                "they moved to the method's output namespace as "
                + ", ".join(_MOVED_OUTPUT_KEYS[key] for key in relocated))
        unknown = set(section) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(
                f"gamd section has unknown keys {sorted(unknown)} "
                f"(known: {sorted(DEFAULT_SETTINGS)})")
        settings.update(section)
        if settings["mode"] not in _VALID_MODES and not settings["channels"] and not settings["boosts"]:
            raise ValueError(
                f"gamd.mode must be one of {_VALID_MODES} (or define "
                f"gamd.channels explicitly), got {settings['mode']!r}")
        sigma0_written = "sigma0" in section
        sigma0 = float(settings["sigma0"]) if sigma0_written else None
        if sigma0 is not None and (not np.isfinite(sigma0) or sigma0 <= 0.0):
            raise ValueError(f"gamd.sigma0 must be > 0, got {sigma0}")
        self.settings = settings
        if settings["energy_unit"] not in ("kJ/mol", "kcal/mol"):
            raise ValueError("gamd.energy_unit must be kJ/mol or kcal/mol")
        self.energy_factor = 4.184 if settings["energy_unit"] == "kcal/mol" else 1.0
        # The built-in default is 6 kcal/mol; a written number is in
        # ``energy_unit`` (kJ/mol unless the section overrides the unit).
        self.sigma0 = (DEFAULT_SIGMA0_KCAL_MOL * 4.184 if sigma0 is None
                       else sigma0 * self.energy_factor)
        if settings["threshold_policy"] not in ("lower", "upper"):
            raise ValueError("gamd.threshold_policy must be lower or upper")

        # -- the per-stage lengths (plan-level steps mapping; ADR-0010) ----
        gamd_steps = dict(getattr(plan, "gamd_steps", None) or {})
        missing = [name for name in ALL_STAGE_NAMES if name not in gamd_steps]
        if missing:
            raise ValueError(
                f"gamd needs the per-stage steps mapping with keys "
                f"{list(ALL_STAGE_NAMES)}; missing {missing}")
        self.gamd_steps = {name: int(gamd_steps[name]) for name in ALL_STAGE_NAMES}
        self.restart_from = settings["restart_from"]
        if self.restart_from is not None and self.restart_from not in ALL_STAGE_NAMES:
            raise ValueError(
                f"gamd.restart_from must be one of {list(ALL_STAGE_NAMES)}, "
                f"got {self.restart_from!r}")
        self.calibration_input = settings["calibration_input"]
        self.calibration_provenance = None

        self.stage = None
        self.exchange = None
        self.parameter_history = {}
        self.channel_settings = {}
        self.partition_report = None
        # -- the method's output namespace (output.gamd; ADR-0011) ---------
        output = _output_settings(plan)
        self.report_tape = bool(output["report"])
        self.sample_interval = int(output["sample_interval"])
        if self.sample_interval < 1:
            raise ValueError("output.gamd.sample_interval must be >= 1")
        self.trace_interval = int(output["trace_interval"])
        if self.trace_interval < 1:
            raise ValueError("output.gamd.trace_interval must be >= 1")

        # -- the channels (mode-driven; explicit channels override) --------
        self.channels: dict[str, tuple[int, ...]] = {}
        if settings["boosts"]:
            if settings["channels"]:
                raise ValueError("gamd.boosts and gamd.channels are mutually exclusive")
            if not provides(kernel, PartitionOps):
                raise NotImplementedError("explicit GaMD boosts require PartitionOps")
            self.partition_report = kernel.compile_boost_partitions(settings["boosts"])
            for row, entry in zip(self.partition_report["channels"], settings["boosts"], strict=True):
                label = str(row["label"])
                self.channels[label] = tuple(row["groups"])
                unit = entry.get("energy_unit", settings["energy_unit"])
                if unit not in ("kJ/mol", "kcal/mol"):
                    raise ValueError(f"gamd.boosts.{label}.energy_unit must be kJ/mol or kcal/mol")
                sigma = (self.sigma0 if "sigma0" not in entry else
                         float(entry["sigma0"]) * (4.184 if unit == "kcal/mol" else 1.0))
                policy = entry.get("threshold_policy", settings["threshold_policy"])
                if not np.isfinite(sigma) or sigma <= 0 or policy not in ("lower", "upper"):
                    raise ValueError(f"invalid calibration settings for boost {label!r}")
                self.channel_settings[label] = {"sigma0": sigma, "threshold_policy": policy}
        elif settings["channels"]:
            for entry in settings["channels"]:
                if "label" not in entry or "groups" not in entry:
                    raise ValueError(
                        "every gamd.channels entry needs 'label' and "
                        f"'groups', got {entry!r}")
                self.channels[str(entry["label"])] = \
                    tuple(int(g) for g in entry["groups"])
        else:
            self.channels["total"] = ()
            if settings["mode"] == "dual":
                discover = getattr(kernel, "torsion_force_groups", None)
                groups = tuple(discover()) if callable(discover) else ()
                if not groups:
                    raise NotImplementedError(
                        "gamd dual boost needs the system's torsion forces "
                        "in their own force group(s): this kernel/system "
                        "reports none (openmm: no torsion forces; fake: "
                        "install a dihedral restraint first)")
                self.channels["dihedral"] = groups

        #: label -> the calibration outcome actually pushed to the kernel
        self.calibration: dict[str, dict] = {}

    # -- entry point --------------------------------------------------------

    def prepare(self):
        """Install zero-strength channels, plan the resume, build the stages.

        Channels are installed at ZERO strength — the same pre-Context
        discipline as ``install_bias`` — and the calibrated (threshold E,
        harmonic k) values are pushed through ``set_boost_param`` at each
        stage boundary (the BiasParamOps pattern applied to channels).  No
        stage retunes the boost while it is running; see ADR-0009/0010 for
        the hand-over chain.  ``mode:
        total`` installs one ``total`` channel (``groups == ()``, the
        system energy); ``mode: dual`` adds a ``dihedral`` channel over
        the system's torsion force groups (discovered through the kernel's
        duck-typed ``torsion_force_groups()``; group ids stay opaque
        ints); ``channels: [{label, groups}]`` defines explicit channels.

        The driver runs the production loop (driver.run_prepared_method)
        with the gamd.tsv probe this method builds; nothing physics-side
        fires on ``on_step`` — the boost is continuous inside the kernel.
        """
        from neomd.driver import PreparedMethod
        from neomd.probes import GamdProbe

        if not provides(self.kernel, BoostOps):
            raise NotImplementedError(
                f"kernel {self.kernel.name!r} does not provide BoostOps "
                "(install_boost / set_boost_param / boost_potentials); "
                "GaMD cannot run on it")

        # zero strength: k = 0 (and a threshold far above any energy) —
        # the same dynamics as an un-boosted run (ADR-0005: the unbiased
        # stages have no bit-string-identity obligations)
        self.kernel.install_boost(
            BoostChannelIR(label=label, groups=groups)
            for label, groups in self.channels.items())

        # resume through the single owner — AFTER install_boost (the
        # openmm Context must come up with the boost integrator in place)
        from neomd.resume import plan_resume

        resume_plan = plan_resume(self.plan, self.kernel, self.sink)
        resumed = resume_plan is not None

        source_gamd = self.sink is not None and self.sink.exists(GamdStages.filename)
        if (resumed and not source_gamd and self.sink is not None
                and any(self.sink.exists(name) for name in
                        (CALIBRATION_FILENAME, TAPE_FILENAME, "gamd_metadata.json"))):
            raise ValueError(
                "continue_md: gamd artifacts found but gamd_stages.json is "
                "missing (pre-ADR-0010 or damaged directory); point "
                "input_files.checkpoint at a checkpoint to adopt the state "
                "into a fresh gamd run instead")
        self.stage = GamdStages(self, resumed=resumed, source_gamd=source_gamd)
        if resume_plan is not None and self.stage.trims:
            # the per-stage tapes are bounded by their OWN leg, so only the
            # method can name them; the ResumePlan the driver reads carries
            # that set (resume.py still owns the trimming mechanics)
            from dataclasses import replace

            resume_plan = replace(resume_plan,
                                  trims={**resume_plan.trims, **self.stage.trims})
        if self.calibration_input:
            self._load_calibration_input()
        if self.calibration:
            # resume/restart or external calibration: push, never re-run
            # (pushing over a checkpoint is idempotent — ADR-0005)
            self._push_calibration()
        # a changed boost-stage layout moves which stage ends the chain:
        # re-derive production's set from that stage's restored samples
        self.stage.reconcile_handoff()

        if self.settings["exchange"] is not None:
            from neomd.methods.ligand_exchange import LigandExchange
            # the ledger's event order must survive the per-stage step reset
            # (ADR-0011): journal on the stage's virtual clock
            self.exchange = LigandExchange(self.kernel, self.settings["exchange"], self.sink,
                                           resumed=resumed, clock=self.stage.virtual_step)
        if self.sink is not None:
            if self.partition_report is not None:
                self.sink.write_bytes("gamd_partitions.json", json.dumps(self.partition_report, indent=2).encode())
            from neomd.driver import timestep_ps
            self.sink.write_bytes("gamd_metadata.json", json.dumps({
                # schema 1 stays the analysis contract (read_gamd pins it);
                # the staged-protocol fields are additive provenance
                "schema_version": 1, "energy_unit": "kJ/mol", "time_unit": "ps",
                "labels": list(self.channels), "frame_semantics": "saved_configuration",
                "timestep_ps": timestep_ps(self.plan), "settings": self.settings,
                "output_gamd": {"report": self.report_tape,
                                "sample_interval": self.sample_interval,
                                "trace_interval": self.trace_interval},
                "stages": self.gamd_steps, "run_mode": self.stage.mode,
                "restart_from": self.restart_from,
                "calibration_input": self.calibration_provenance,
                "partition_report": self.partition_report,
                "exchange_convention": "post_exchange" if self.exchange is not None else None,
                "exchange_selection_policy": None if self.exchange is None else self.exchange.selection_policy,
                "sampling_status": "experimental_nonequilibrium_exchange" if self.exchange is not None else "fixed_bias",
                "equilibrium_reweighting_qualified": self.exchange is None,
            }, indent=2).encode())
        tapes: dict = {}
        if self.sink is not None:
            tapes[TAPE_FILENAME] = GamdProbe(
                self.sink,
                interval=self.trace_interval,
                labels=list(self.channels),
                append=resumed and TAPE_FILENAME in resume_plan.trims,
                metadata=self.metadata,
                trajectory_interval=getattr(self.plan, "trajectory_interval", 0),
                dt_ps=timestep_ps(self.plan),
            )
        return PreparedMethod(
            on_step=None,
            fgroups={},  # GaMD installs no bias force
            resume_plan=resume_plan,
            tapes=tapes,
            finish=self._finish,
            preparation=self.stage,
            before_step=None if self.exchange is None else self.exchange.on_step,
            before_step_interval=1 if self.exchange is None else self.exchange.interval,
            checkpoint=self.checkpoint,
            total_steps=self.stage.total_steps,
        )

    def metadata(self):
        return {"phase": self.stage.phase if self.stage is not None else "production",
                "parameter_revision": max((entry["revision"] for entry in
                                           self.parameter_history.values()), default=0),
                "slot_to_identity": [] if self.exchange is None else list(self.exchange.slot_to_identity)}

    def checkpoint(self):
        if self.stage is not None:
            self.stage.persist()
        if self.exchange is not None:
            self.exchange.persist()

    def next_exchange_boundary(self, target):
        if self.exchange is not None:
            interval = self.exchange.interval
            return min(target, (self.kernel.current_step // interval + 1) * interval)
        return target

    def exchange_at_boundary(self):
        if self.exchange is not None and self.kernel.current_step % self.exchange.interval == 0:
            self.exchange.on_step(self.kernel.current_step)

    def _finish(self, result) -> MethodResult:
        """End-of-run artifacts + the result drive() records."""
        from neomd.driver import CHECKPOINT_FILENAME

        if self.sink is not None:
            self.sink.write_bytes(CHECKPOINT_FILENAME, self.kernel.snapshot())
        mean_boost = {}
        if self.sink is not None:
            try:
                _steps, boost, _scale = read_gamd_trace(
                    self.sink.path(TAPE_FILENAME))
                mean_boost = {label: float(np.mean(values))
                              for label, values in boost.items()}
            except (FileNotFoundError, NotImplementedError):
                pass
            except ValueError as error:
                if "has no data rows" not in str(error):
                    raise
                # A valid run can finish exactly at the production boundary.
        return MethodResult(
            steps_done=result.steps_done,
            channels={label: {"threshold": params["threshold"],
                              "k": params["k"]}
                      for label, params in self.calibration.items()},
            mean_boost=mean_boost,
            positions_sha256=result.positions_sha256,
        )

    def run(self) -> MethodResult:
        """Direct-construction entry: prepare + the driver's method loop."""
        from neomd.driver import run_prepared_method

        return run_prepared_method(self.kernel, self.plan, self.prepare(),
                                   sink=self.sink, logger=self.log)

    # -- calibration (parameter selection + persistence) ---------------------

    def _record_parameters(self, stage, calibration, *, extra=None) -> dict:
        """Store one parameter set under its statistics stage (ADR-0010).

        The journal's ``parameter_history`` is keyed by the stage that
        produced the set, so only a stage that actually acted as a source
        ever appears; each entry carries a monotone ``revision`` (the number
        the gamd.tsv rows report) so the mapping stays explicit even though
        the container is a mapping.
        """
        revision = max((entry["revision"] for entry in
                        self.parameter_history.values()), default=0) + 1
        entry = {"revision": revision, "step": int(self.kernel.current_step),
                 "channels": calibration}
        if extra:
            entry.update(extra)
        self.parameter_history[stage] = entry
        return entry

    def complete_calibration(self, samples, *, moments=None, persist=True, stage=None):
        """Select and persist calibrated parameters from completed cMD samples.

        ``stage`` names the statistics stage that produced this set; the
        journal keys ``parameter_history`` by it (ADR-0010), so only a stage
        that actually acted as a source ever appears.
        """
        calibration = {}
        for label, values in samples.items():
            sigma0 = self.channel_settings.get(label, {}).get("sigma0", self.sigma0)
            policy = self.channel_settings.get(label, {}).get("threshold_policy", self.settings["threshold_policy"])
            count = len(values) if moments is None else moments[label]["count"]
            if moments is None:
                params = _channel_params(np.asarray(values, dtype=np.float64), sigma0, policy)
            else:
                m = moments[label]
                params = _params_from_statistics(m["min"], m["max"], m["mean"],
                                                 float(np.sqrt(max(0., m["m2"] / count))),
                                                 sigma0, policy)
            calibration[label] = params
            if params["k"] == 0.0:
                log_disabled = self.log.warning if label not in self.calibration else self.log.debug
                log_disabled(
                    "gamd channel %r kept at zero strength (sigmaV=%.6g "
                    "kJ/mol — the energy did not vary enough to calibrate)",
                    label, params["sigma"])
            else:
                log_calibrated = self.log.info if not self.calibration else self.log.debug
                log_calibrated(
                    "gamd channel %r calibrated (%s bound): E=%.6g kJ/mol, "
                    "k0*k=%.6g 1/(kJ/mol) from %d samples "
                    "(Vmax=%.6g Vmin=%.6g sigmaV=%.6g)",
                    label, params["bound"], params["threshold"],
                    params["k"], count, params["vmax"],
                    params["vmin"], params["sigma"])
        self.calibration = calibration
        self._record_parameters(stage if stage is not None else "external",
                                calibration)
        if persist:
            self._persist_calibration()
        self._push_calibration()

    def _load_calibration_input(self) -> None:
        """Adopt or recompute boost parameters from ``gamd.calibration_input``.

        Two accepted forms (the file is external provenance — its sha256
        lands in gamd_calibration.json and gamd_metadata.json):

        * a ``gamd_calibration.json`` — its per-channel (threshold, k) are
          adopted verbatim (a restart's pinned parameters);
        * an energy-sample tape (``gamd.tsv`` / ``state.tsv``) — Vmin/Vmax/
          σV statistics are recomputed and (E, k) re-selected through the
          usual literature rules.
        """
        path = self.calibration_input
        data = Path(path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if path.endswith(".json"):
            payload = json.loads(data.decode("utf-8"))
            channels = payload["channels"]
            missing = set(self.channels) - set(channels)
            if missing:
                raise ValueError(
                    f"calibration_input {path!r} lacks channel(s) "
                    f"{sorted(missing)} (found {sorted(channels)})")
            self.calibration = {label: channels[label] for label in self.channels}
            kind = "calibration_json"
        else:
            samples = _tape_energy_samples(path, self.channels)
            calibration = {}
            for label, values in samples.items():
                sigma0 = self.channel_settings.get(label, {}).get("sigma0", self.sigma0)
                policy = self.channel_settings.get(label, {}).get(
                    "threshold_policy", self.settings["threshold_policy"])
                calibration[label] = _channel_params(values, sigma0, policy)
            self.calibration = calibration
            kind = "tape"
        self.calibration_provenance = {"path": str(path), "sha256": sha, "kind": kind}
        self._record_parameters("external", self.calibration,
                                extra={"source": self.calibration_provenance})
        self._persist_calibration()
        self.log.info("gamd parameters %s from %s (sha256 %.12s)",
                      "adopted" if kind == "calibration_json" else "recomputed",
                      path, sha)

    def _persist_calibration(self):
        if self.sink is not None and self.calibration:
            self.sink.write_bytes(CALIBRATION_FILENAME, json.dumps({
                "energy_unit": "kJ/mol", "channels": self.calibration,
                "parameter_history": self.parameter_history,
                "provenance": self.calibration_provenance,
            }, indent=2, sort_keys=True).encode("utf-8"))

    def _push_calibration(self) -> None:
        """Push (threshold, k) per channel via ``set_boost_param``; pushing
        over a restored checkpoint is idempotent (resume re-pushes, it does
        not re-calibrate)."""
        for label, params in self.calibration.items():
            self.kernel.set_boost_param(label, "threshold", params["threshold"])
            self.kernel.set_boost_param(label, "k", params["k"])

    def adopt_calibration(self, calibration, *, record=True, stage=None) -> None:
        """Set the live channel parameters without re-running statistics.

        Used when a resume changes which stage sits at the end of the
        calibration chain: the set is re-derived from the source stage's
        restored samples and recorded as a new parameter revision.
        """
        self.calibration = calibration
        if record:
            self._record_parameters(stage if stage is not None else "external",
                                    calibration)
        self._push_calibration()


def _layout_description(settings) -> dict:
    """What the boost channels ARE: the selection structure, not the strengths.

    A continue may retune ``sigma0`` / ``threshold_policy`` / ``energy_unit``
    (they only shape the parameters a LATER calibration derives), but it may
    not change which interactions a channel owns: the restored Context's force
    structure and the journal's pinned parameters (keyed by channel label)
    belong to that layout.  Indices are treated as sets and force types as a
    set too, so re-spelling the same selection is not drift.
    """
    boosts = settings.get("boosts")
    if boosts:
        return {"boosts": [
            {"label": boost.get("label"),
             "remaining": bool(boost.get("remaining")),
             "selectors": [
                 {"atom_indices": sorted(int(a) for a in (selector.get("atom_indices") or [])),
                  "force_types": sorted(_force_type_list(selector.get("force_types"))),
                  "include_boundary": {str(k): bool(v) for k, v in
                                       sorted((selector.get("include_boundary") or {}).items())}}
                 for selector in (boost.get("selectors") or [])]}
            for boost in boosts]}
    channels = settings.get("channels")
    if channels:
        return {"channels": [{"label": row.get("label"),
                              "groups": sorted(int(g) for g in (row.get("groups") or []))}
                             for row in channels]}
    return {"mode": settings.get("mode")}


def _force_type_list(types):
    """A selector's force types as an iterable (the ``all`` alias stays literal)."""
    if types is None:
        return []
    if isinstance(types, str):
        return [types]
    return list(types)


def _recorded_payload(run, saved):
    """The configuration a run recorded, or None when it kept none.

    Newer journals carry the fingerprint payload; older ones are described by
    their sibling ``gamd_metadata.json`` (the settings/output namespace that
    run actually used), which is what makes both the layout gate and the
    drift warning work for directories written before the payload existed.
    """
    payload = saved.get("fingerprint_payload")
    if isinstance(payload, dict) and isinstance(payload.get("settings"), dict):
        return payload
    if run.sink is not None and run.sink.exists("gamd_metadata.json"):
        try:
            metadata = json.loads(run.sink.read_bytes("gamd_metadata.json"))
        except (ValueError, TypeError, UnicodeDecodeError):
            return None
        settings = metadata.get("settings")
        if isinstance(settings, dict):
            recorded = {"settings": {
                key: value for key, value in settings.items()
                if key not in ("restart_from", "calibration_input")}}
            output = metadata.get("output_gamd")
            if isinstance(output, dict):
                recorded["output"] = {
                    "sample_interval": output.get("sample_interval"),
                    "trace_interval": output.get("trace_interval")}
            return recorded
    return None


def _recorded_layout(run, saved):
    """The recorded boost layout, or None when the directory did not keep it."""
    payload = _recorded_payload(run, saved)
    if payload is None:
        return None
    layout = payload.get("layout")
    if isinstance(layout, dict):
        return layout
    settings = payload.get("settings")
    return _layout_description(settings) if isinstance(settings, dict) else None


def _layout_drift(run, saved) -> str:
    """Describe a changed boost layout ("" when unchanged or unrecorded)."""
    recorded = _recorded_layout(run, saved)
    if recorded is None:
        return ""  # nothing to compare against; the restore itself is the guard
    current = _layout_description(run.settings)
    if recorded == current:
        return ""
    return f"recorded {_short(recorded)} -> now {_short(current)}"


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True, default=list)
    return text if len(text) <= 60 else text[:57] + "..."


def _fingerprint_drift(recorded, current) -> str:
    """Name the keys that differ between a journal and this run's config.

    Best effort: journals written before the payload was recorded return "".
    The comparison is per top-level group (``settings.*`` / ``output.*`` plus
    the resolved channel layout), which is exactly what the hash covers.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return ""
    # normalize through JSON: a journal round-trips tuples to lists, and the
    # comparison must not report that as a change
    recorded = json.loads(json.dumps(recorded, sort_keys=True, default=list))
    current = json.loads(json.dumps(current, sort_keys=True, default=list))
    drifted = []
    for group in ("settings", "output"):
        before = recorded.get(group) or {}
        after = current.get(group) or {}
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                drifted.append(f"{group}.{key}: {_short(before.get(key))} -> "
                               f"{_short(after.get(key))}")
    if "channels" in recorded and "channels" in current \
            and recorded["channels"] != current["channels"]:
        drifted.append("channels (boost labels / force-group layout) changed")
    return ("; drifted: " + "; ".join(drifted)) if drifted else ""


class GamdStages:
    """Per-leg staged preparation: one independent integration leg per stage.

    ADR-0011 replaces the absolute step axis with LEGS.  A leg is one
    contiguous stretch of integration with its own Context and its own step
    axis starting at 0; ``configured`` holds the per-stage lengths, and the
    kernel's ``current_step`` is always "how far into the CURRENT leg".  This
    object owns the four kernel-driven legs (``unbiased_eq`` →
    ``biased_eq2``) and hands the fifth — ``production`` — to the driver's
    loop; entering it (``begin_next_leg``) is what makes :attr:`done` true.

    Crossing a leg boundary calls ``kernel.start_leg(index)``: leg 0 is the
    Context the run already started from, every later leg is a freshly built
    Context under the derived seed (``spec.seed + index``), which is what
    makes each leg's random stream independent and reproducible.  Positions,
    velocities and box carry over; the RNG and the volume-proposal state
    (native or adapter-owned) do not.  Consequently there is NO global axis anywhere:
    ``gamd.restart_from`` re-enters its leg at 0 and an adopted external
    checkpoint has its step counter zeroed rather than continued.

    Every leg transition persists a boundary checkpoint naming the state
    that BEGINS a stage — ``ckpt/entry.ckpt`` for ``unbiased_eq``,
    ``ckpt/<previous stage>.ckpt`` afterwards (zero-length stages share
    their successor's).  Those are the restore points ``restart_from``
    re-enters.  ``completed`` (journal, schema 4) holds each stage's already
    run step count; a resumed mid-leg checkpoint restores its leg's own
    counter.  Adaptive statistics start a new cumulative window per source
    stage after boost preparation; production never updates parameters.
    """

    filename = "gamd_stages.json"
    #: 4: per-leg accounting — no absolute ``stage_ends``/``start_step``/
    #: ``final_step``; ``completed`` is per leg and ``phase`` names the leg the
    #: restored Context belongs to (ADR-0011)
    _SCHEMA_VERSION = 4

    def __init__(self, run, *, resumed=False, source_gamd=False):
        self.run = run
        settings = run.settings
        self.configured = dict(run.gamd_steps)
        if any(n < 0 for n in self.configured.values()):
            raise ValueError("GaMD stage lengths must be nonnegative")
        # ONE cadence for every equilibration leg: the per-stage V(P)
        # statistics that calibrate the next boost stage
        self.interval = int(run.sample_interval)
        if self.interval < 1:
            raise ValueError("output.gamd.sample_interval must be positive")
        # fingerprint: PHYSICAL configuration only — stage lengths and
        # continue switches (restart_from / calibration_input) may change
        # between runs; the boost layout and sampling cadence may not.
        #: the exact object :attr:`fingerprint` hashes.  The journal keeps a
        #: copy of it, so a mismatch can NAME the drifted keys instead of only
        #: refusing — an edited/moved YAML key is the common cause and the hash
        #: alone cannot say which one it was.
        self.fingerprint_payload = {
            "settings": {key: settings[key] for key in sorted(settings)
                         if key not in ("restart_from", "calibration_input")},
            # the sampling cadence changes the statistics (and therefore the
            # boost), so it is physical; the tape cadence only decides how
            # many rows are written.  ``report`` is a display switch and is
            # deliberately NOT here.
            "output": {"sample_interval": run.sample_interval,
                       "trace_interval": run.trace_interval},
            "channels": list(run.channels.items()),
            # the selection structure on its own, so a later continue can tell
            # a retuned parameter from a changed boost layout (see _layout_drift)
            "layout": _layout_description(settings),
        }
        self.fingerprint = hashlib.sha256(json.dumps(
            self.fingerprint_payload, sort_keys=True,
            default=list).encode()).hexdigest()
        self.records: list[dict] = []
        self._last_flush = 0
        self._stepped = 0
        #: per-stage already-run steps; earlier legs are always ``configured``
        self.completed = {name: 0 for name in ALL_STAGE_NAMES}
        #: index of the CURRENT leg (``len(ALL_STAGE_NAMES) - 1`` = production)
        self.index = 0
        #: has the current leg's Context been established (start_leg run)?
        self._begun = False
        #: this session's basis inside the current leg: the kernel step it
        #: started at and the completed count that step corresponded to
        self._active_step = 0
        self._active_base = 0
        self.mode = "fresh"
        #: the layout the journal recorded (resume only): a changed
        #: boost-stage chain re-derives production's set (reconcile_handoff)
        self._saved_configured = None

        if resumed and source_gamd and run.restart_from is None:
            self.mode = "resume"
            saved = self._load_saved(run)
            self._adopt_saved_progress(run, saved)
            self._restore_history(run, saved)
        elif resumed and source_gamd:
            self.mode = "restart"
            saved = self._load_saved(run)
            self.index = ALL_STAGE_NAMES.index(run.restart_from)
            boundary = self._boundary_name(self.index)
            if run.sink is None or not run.sink.exists(boundary):
                raise ValueError(
                    f"gamd.restart_from {run.restart_from!r} needs the "
                    f"{boundary} boundary checkpoint, which does not exist "
                    "(the stage before it never completed); available "
                    f"boundaries: {self._available_boundaries(run) or 'none'}")
            # the rolling checkpoint restored by plan_resume is the WRONG
            # state for a rerun — swap in the stage-boundary state and
            # re-trim every tape to its per-leg bound (resume stays the
            # trimming owner)
            run.kernel.restore(run.sink.read_bytes(boundary))
            recorded = self._checkpoint_leg(run)
            if recorded is not None and recorded != self.index:
                raise ValueError(
                    f"gamd.restart_from {run.restart_from!r} restored "
                    f"{boundary}, whose recorded leg is "
                    f"{ALL_STAGE_NAMES[recorded]!r}: the boundary checkpoint "
                    "and the requested stage disagree")
            for i, name in enumerate(ALL_STAGE_NAMES):
                self.completed[name] = self.configured[name] if i < self.index else 0
            self._restore_history(run, saved)
            if self.index >= 2 and not run.calibration and not run.calibration_input:
                raise ValueError(
                    f"gamd.restart_from {run.restart_from!r} reruns biased "
                    "stages but no calibration exists at the boundary: "
                    "provide gamd.calibration_input pointing at the source "
                    "run's calibration file")
        elif resumed:
            # adopt: a non-gamd checkpoint (external equilibration) — or a
            # boundary checkpoint for a new-path restart_from
            if run.restart_from not in (None, "unbiased_eq"):
                if not run.calibration_input:
                    raise ValueError(
                        f"gamd.restart_from {run.restart_from!r} from a "
                        "non-gamd path needs gamd.calibration_input pointing "
                        "at the source run's calibration file")
                self.mode = "restart"
                self.index = ALL_STAGE_NAMES.index(run.restart_from)
                for i, name in enumerate(ALL_STAGE_NAMES):
                    self.completed[name] = self.configured[name] if i < self.index else 0
            else:
                self.mode = "adopt"

        self.moments = {phase: {} for phase in _SAMPLE_PHASES}
        for record in self.records:
            self._accumulate(record)
        #: artifact -> bound for the per-stage tapes this resume adopted;
        #: ``GamdRun.prepare`` folds it into the ResumePlan the driver reads,
        #: so every probe's append decision agrees with the trim that happened
        self.trims: dict[str, int] = {}
        if resumed and run.sink is not None:
            # resume owns trimming: the per-leg bounds need the leg the
            # restored Context belongs to, which only this object knows
            from neomd.resume import trim_to

            self.trims = trim_to(
                run.sink,
                self._trim_bounds(self.configured, self.index, self.completed),
                run.plan)
        # adopt/restart inherit a Context that is NOT their leg's own: enter
        # the leg now (step 0, derived seed, entry checkpoint).  A fresh run
        # and a mid-leg resume already are inside their leg.
        self._sync_checkpoint_metadata()  # before any snapshot below
        if self.mode in ("adopt", "restart"):
            self._enter()
        elif run.sink is not None:
            # a fresh run whose first legs are zero-length still needs its
            # entry/boundary checkpoints before any step is taken
            self._write_zero_length_boundaries()
        self.persist()

    # -- leg bookkeeping -----------------------------------------------------

    def _adopt_saved_progress(self, run, saved):
        """Restore per-leg progress from the checkpoint's leg record.

        The restored Context's leg is state, not configuration: its clock is
        leg-relative and its RNG stream is that leg's, so it travels WITH the
        checkpoint (``_LEG_METADATA_KEY``) rather than being inferred from the
        journal — the journal may run ahead of the checkpoint (flushed after
        it) or behind it (production persists only at checkpoint probes), and
        a clock of 0 is ambiguous between "this leg's start" and "the next
        leg's start" without it.
        """
        index = self._checkpoint_leg(run)
        if index is None:
            # a checkpoint from before the leg record existed: the journal's
            # phase is the only hint, and it can be ahead of the state
            index = ALL_STAGE_NAMES.index(saved["phase"])
            run.log.warning(
                "gamd: restored checkpoint carries no stage-leg record; "
                "assuming the journal's stage %r (positions captured at the "
                "first step that mismatch a stage change cannot be recovered)",
                saved["phase"])
        leg_step = int(run.kernel.current_step)
        if self.configured[ALL_STAGE_NAMES[index]] == 0:
            # the recorded leg is zero-length under the CURRENT stage lengths;
            # its clock says nothing about the leg that follows it
            while (index < len(ALL_STAGE_NAMES) - 1
                   and self.configured[ALL_STAGE_NAMES[index]] == 0):
                index += 1
            leg_step = 0
        self.index = index
        for i, name in enumerate(ALL_STAGE_NAMES):
            self.completed[name] = self.configured[name] if i < index else 0
        self.completed[ALL_STAGE_NAMES[index]] = min(
            self.configured[ALL_STAGE_NAMES[index]], max(0, leg_step))
        self._begun = True
        self._active_step = int(run.kernel.current_step)
        self._active_base = self.completed[ALL_STAGE_NAMES[index]]
        self._saved_configured = dict(saved.get("configured") or self.configured)

    # -- the leg record (checkpoint metadata) --------------------------------

    @staticmethod
    def _checkpoint_leg(run):
        """The leg index recorded in the restored checkpoint, or None."""
        from neomd.kernel.port import CheckpointMetadataOps, provides

        if not provides(run.kernel, CheckpointMetadataOps):
            return None
        metadata = run.kernel.checkpoint_metadata() or {}
        leg = metadata.get(_LEG_METADATA_KEY)
        if isinstance(leg, bool) or not isinstance(leg, int) \
                or not 0 <= leg < len(ALL_STAGE_NAMES):
            return None
        return leg

    def _sync_checkpoint_metadata(self):
        """Commit the running leg into every subsequent checkpoint snapshot."""
        from neomd.kernel.port import CheckpointMetadataOps, provides

        kernel = self.run.kernel
        if not provides(kernel, CheckpointMetadataOps):
            return
        metadata = dict(kernel.checkpoint_metadata() or {})
        metadata[_LEG_METADATA_KEY] = int(self.index)
        kernel.set_checkpoint_metadata(metadata)

    def _begin_leg(self):
        """Establish the current leg: step axis 0, derived RNG, entry ckpt."""
        from neomd.kernel.port import LegOps, provides

        if not provides(self.run.kernel, LegOps):
            raise NotImplementedError(
                f"kernel {self.run.kernel.name!r} does not provide LegOps "
                "(start_leg); staged GaMD needs one Context per stage")
        self.run.kernel.start_leg(self.index)
        self._begun = True
        self._active_step = 0
        self._active_base = 0
        self.completed[self.phase] = 0
        self._sync_checkpoint_metadata()
        self._write_boundary(self.index)

    def _enter(self):
        """Walk past zero-length legs, then :meth:`_begin_leg`."""
        if self._begun:
            return
        while (not self.done) and self.configured[self.phase] == 0:
            self._write_boundary(self.index)
            self.index += 1
        self._begin_leg()

    def _write_zero_length_boundaries(self):
        """Entry/zero-length boundaries of a not-yet-started fresh run."""
        sink = self.run.sink
        if sink is None:
            return
        index = 0
        while index < len(ALL_STAGE_NAMES) - 1 and self.configured[ALL_STAGE_NAMES[index]] == 0:
            self._write_boundary(index)
            index += 1
        if index == 0:
            self._write_boundary(0)

    def begin_next_leg(self):
        """Close the current leg and enter the next one (driver-driven).

        Called by the driver AFTER the closing stage's boundary observation,
        so the tapes of the stage that just ended are written under its own
        name and its own last step number.  The journal is flushed here, in
        the same order as every other flush: never with a ``phase`` that
        disagrees with the Context the checkpoint would carry.
        """
        if self.done:
            raise RuntimeError("GaMD production is the last leg; none follows")
        name = self.phase
        self.completed[name] = self.configured[name]
        self.index += 1
        # the next leg needs its OWN Context: ``_enter`` short-circuits while
        # the live Context already belongs to the current leg, which it no
        # longer does after this transition
        self._begun = False
        self._enter()
        self.persist()

    @property
    def leg_end(self):
        """The current leg's length: its kernel clock ends there."""
        return int(self.configured[self.phase])

    @property
    def total_steps(self):
        """Production leg length (the driver's per-leg stepper target).

        Legs all restart at 0, so the target is the leg's own length, not a
        remainder against a global axis.
        """
        return int(self.configured["production"])

    # -- continue helpers ----------------------------------------------------

    @staticmethod
    def _boundary_name(index):
        """Boundary checkpoint preceding stage ``index`` ("entry" for 0)."""
        return f"ckpt/{'entry' if index == 0 else ALL_STAGE_NAMES[index - 1]}.ckpt"

    def _available_boundaries(self, run):
        names = [self._boundary_name(i) for i in range(len(ALL_STAGE_NAMES))]
        return [name for name in names if run.sink is not None and run.sink.exists(name)]

    def _load_saved(self, run):
        if run.sink is None or not run.sink.exists(self.filename):
            raise FileNotFoundError("staged GaMD resume requires gamd_stages.json")
        saved = json.loads(run.sink.read_bytes(self.filename))
        if saved.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError(
                "gamd_stages.json is from an incompatible version "
                f"(schema {saved.get('schema_version')}, need "
                f"{self._SCHEMA_VERSION}); adopt the directory's "
                "output.ckpt into a fresh run via input_files.checkpoint "
                "instead")
        if saved["fingerprint"] != self.fingerprint:
            # A continue may deliberately change physical settings (sigma0,
            # threshold policy, unit, sampling/tape cadence, exchange): they
            # shape only what the run does FROM HERE, and the restored Context
            # keeps the parameters its leg started with.  What it may NOT
            # change is the boost LAYOUT — the resolved channels are the
            # System's force structure the restored state belongs to, and the
            # journal's pinned parameters are keyed by their labels.
            drift = _layout_drift(run, saved)
            if drift:
                raise ValueError(
                    "GaMD ordered configuration changed since checkpoint: the "
                    "boost layout differs (" + drift + "); a continue must "
                    "keep the same channels — adopt the run's output.ckpt into "
                    "a fresh run (input_files.checkpoint) to change the layout")
            run.log.warning(
                "gamd: continuing with a configuration changed since the "
                "checkpoint (allowed — the boost layout is unchanged)%s",
                _fingerprint_drift(_recorded_payload(run, saved),
                                   self.fingerprint_payload))
        return saved

    @staticmethod
    def _trim_bounds(configured, index, completed):
        """Per-artifact trim bound for ``resume.trim_to`` (ADR-0011).

        Every staged tape carries ONE leg's rows numbered from 0, so a single
        global bound is wrong: trimming ``unbiased_eq.state`` (1000 rows) to a
        later leg's step 30 would delete data.  A finished leg keeps its full
        length, the leg being resumed keeps its completed count, and a leg
        that has not run keeps nothing (its file will be rebuilt).
        """
        bounds = {}
        for i, name in enumerate(ALL_STAGE_NAMES):
            if i < index:
                bound = configured[name]
            else:
                bound = (completed or {}).get(name, 0)
            for filename in (f"{name}.state", f"{name}.dcd",
                             f"{name}_restraint.tsv"):
                bounds[filename] = int(bound)
        # gamd.tsv belongs to the production leg; it holds nothing before then
        bounds[TAPE_FILENAME] = int((completed or {}).get("production", 0))
        return bounds

    def virtual_step(self, step=None):
        """Monotone run-wide step for artifacts that must not restart.

        Legs have no public global axis, but the ligand-exchange ledger's
        event order (and its pending-transaction guard) has to increase
        strictly across stages: this is the finished legs' total plus the
        current leg's clock.  It is journal bookkeeping only — no user-facing
        tape or step column exposes it.
        """
        step = int(self.run.kernel.current_step) if step is None else int(step)
        return sum(self.configured[name]
                   for name in ALL_STAGE_NAMES[:self.index]) + step

    def _restore_history(self, run, saved):
        """Rebuild the Welford windows from the journal's per-stage samples.

        Records are grouped by the leg that produced them and each leg's
        steps restart at 0, so the replay order is (leg index, step) — never
        a single global step ordering.  A stage that has not finished
        contributes only the rows the restored Context had already produced.
        """
        self.records = sorted(
            ({"step": int(record["step"]), "phase": stage,
              "energies": record["energies"]}
             for stage, records in saved["samples"].items()
             for record in records
             if self._sample_kept(stage, int(record["step"]))),
            key=lambda record: (ALL_STAGE_NAMES.index(record["phase"]),
                                record["step"]))
        run.parameter_history = {
            stage: entry for stage, entry in saved["parameter_history"].items()
            if ALL_STAGE_NAMES.index(stage) < self.index
            or (ALL_STAGE_NAMES.index(stage) == self.index
                and self.completed[self.phase] >= self.configured[self.phase])}
        if run.parameter_history:
            run.calibration = max(
                run.parameter_history.values(),
                key=lambda entry: entry["revision"])["channels"]

    def _sample_kept(self, stage, step):
        """May ``stage``'s sample at ``step`` be replayed on this resume?"""
        if stage not in _SAMPLE_PHASES:
            return False
        position = ALL_STAGE_NAMES.index(stage)
        if position > self.index:
            return False  # a leg that has not run yet has no samples
        return step <= self.completed[stage]

    # -- boundaries ----------------------------------------------------------

    def _write_boundary(self, index):
        """Persist the checkpoint that BEGINS stage ``index``.

        Written while the kernel is exactly at the leg's entry (step 0 of
        that leg), so restoring the blob and re-entering the leg is the same
        operation the uninterrupted run performed — deterministically.
        """
        sink = self.run.sink
        if sink is None:
            return
        name = self._boundary_name(index)
        if not sink.exists(name):
            sink.write_bytes(name, self.run.kernel.snapshot())

    # -- stage protocol --------------------------------------------------------

    @property
    def phase(self):
        """The leg the kernel's clock belongs to (leg 4 = production)."""
        return ALL_STAGE_NAMES[self.index]

    @property
    def done(self):
        # the four kernel-driven legs are done once PRODUCTION has been
        # entered; that leg is the DRIVER's loop, not advance()'s
        return self.index == len(ALL_STAGE_NAMES) - 1

    def _accumulate(self, record):
        moments = self.moments[record["phase"]]
        for label, value in record["energies"].items():
            m = moments.setdefault(label, {"count": 0, "mean": 0., "m2": 0., "min": value, "max": value})
            m["count"] += 1
            delta = value - m["mean"]
            m["mean"] += delta / m["count"]
            m["m2"] += delta * (value - m["mean"])
            m["min"], m["max"] = min(m["min"], value), max(m["max"], value)

    # -- stage-boundary parameter handover -----------------------------------

    def _handoff_source(self, phase):
        """The stage ``phase``'s samples calibrate, or ``None``.

        ``unbiased_production`` always seeds ``biased_eq``; ``biased_eq``
        seeds ``biased_eq2`` only when that stage actually has steps.  The
        last boost stage (``biased_eq2`` with steps, else ``biased_eq``) never
        recalibrates, so ``production`` keeps the parameter set it started
        with — and therefore runs the same boost as that equilibration.
        """
        target = _HANDOFF_SOURCE.get(phase)
        if target is None:
            return None
        if target == "biased_eq2" and self.configured["biased_eq2"] <= 0:
            return None
        return phase

    def _final_source(self, configured):
        """The stage whose statistics define the last boost stage's set."""
        return "biased_eq" if configured.get("biased_eq2", 0) > 0 else "unbiased_production"

    def _handoff_parameters(self, source):
        """(E, k) per channel from one source stage's cumulative moments."""
        calibration = {}
        for label, moment in self.moments[source].items():
            settings = self.run.channel_settings.get(label, {})
            sigma0 = settings.get("sigma0", self.run.sigma0)
            policy = settings.get("threshold_policy", self.run.settings["threshold_policy"])
            calibration[label] = _params_from_statistics(
                moment["min"], moment["max"], moment["mean"],
                float(np.sqrt(max(0., moment["m2"] / moment["count"]))),
                sigma0, policy)
        return calibration

    def _calibrate_handoff(self, phase):
        """Hand one stage's samples to the next boost stage, at its end only.

        An external ``gamd.calibration_input`` pins every channel, so the
        internal statistics are recorded but never recomputed (ADR-0010).
        The un-boosted statistics stage calibrates unconditionally (on a fresh
        run there is no incoming set); a boosted source with a single sample or
        no usable spread keeps the incoming set instead of disabling the
        channel outright.
        """
        if self.run.calibration_input is not None:
            return
        source = self._handoff_source(phase)
        if source is None:
            return
        moments = self.moments[source]
        if not moments:
            # NOT silent: no sample means the stage was shorter than
            # output.gamd.sample_interval, so the next leg keeps whatever
            # (possibly zero) strength is installed.  A production run under
            # zero-strength boosts is a scientific no-op, so say so loudly.
            self.run.log.warning(
                "gamd %s produced no V(P) statistics samples (its leg is "
                "shorter than output.gamd.sample_interval=%d): the next stage "
                "keeps the currently installed boost parameters, which means "
                "ZERO strength on a fresh run; lengthen the stage or lower "
                "output.gamd.sample_interval",
                source, self.run.sample_interval)
            return
        if source == "biased_eq":
            fit = self._handoff_parameters(source)
            if (min(moment["count"] for moment in moments.values()) < 2
                    or any(params["disabled_reason"] for params in fit.values())):
                self.run.log.info(
                    "gamd %s sampled too little spread to re-calibrate; keeping "
                    "the incoming boost parameters for the next stage", source)
                return
        self.run.complete_calibration(
            {label: () for label in moments}, moments=moments, persist=False,
            stage=source)

    def reconcile_handoff(self):
        """Keep production's parameters tied to its last boost equilibration.

        An unchanged configuration needs nothing: the journal's last entry
        already is the set the last boost stage started with.  Only when the
        saved layout put a *different* stage at the end of the chain (e.g. a
        resume that adds or drops ``biased_eq2``) is the set re-derived from
        the new source stage's restored moments, so production cannot inherit
        parameters from a stage layout that no longer exists.
        """
        saved = self._saved_configured
        if saved is None or self.run.calibration_input is not None:
            return
        source = self._final_source(self.configured)
        if source == self._final_source(saved):
            return
        if self._completed_now()[source] < self.configured[source]:
            return  # still to run: advance() hands over at its end
        if not self.moments[source]:
            return
        calibration = self._handoff_parameters(source)
        if calibration == self.run.calibration:
            return
        self.run.log.info(
            "gamd: production boost parameters re-derived from %s after the "
            "boost-stage layout changed", source)
        self.run.adopt_calibration(calibration, stage=source)

    def _completed_now(self):
        """Per-stage completed steps, each counted inside its OWN leg.

        Earlier legs are finished (``configured``); the current leg adds what
        this session ran on top of the count its Context was restored with.
        Legs that have not been entered yet are 0.
        """
        now = int(self.run.kernel.current_step)
        out = dict(self.completed)
        if self._begun:
            name = self.phase
            out[name] = min(self.configured[name],
                            self._active_base + max(0, now - self._active_step))
        return out

    def persist(self):
        if self.run.sink is not None:
            self.run._persist_calibration()
            self.run.sink.write_bytes(self.filename, json.dumps({
                "schema_version": self._SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                # names the drifted keys when a later run refuses to continue
                "fingerprint_payload": self.fingerprint_payload,
                "mode": self.mode,
                "configured": self.configured,
                "completed": self._completed_now(),
                "phase": self.phase, "step": int(self.run.kernel.current_step),
                "samples": {stage: [{"step": record["step"],
                                     "energies": record["energies"]}
                                    for record in self.records
                                    if record["phase"] == stage]
                            for stage in ALL_STAGE_NAMES},
                "sample_counts": {phase: sum(r["phase"] == phase for r in self.records)
                                  for phase in _SAMPLE_PHASES},
                "parameter_history": self.run.parameter_history,
                "current_parameters": self.run.calibration,
            }, sort_keys=True).encode())

    def advance(self, max_steps):
        """Run up to ``max_steps`` steps of the CURRENT leg, never past it.

        Stopping exactly at the leg's last step is what lets the driver write
        that stage's own artifact rows under its own name and step number
        before :meth:`begin_next_leg` swaps in the next Context.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._enter()
        kernel = self.run.kernel
        # A caller that keeps advancing past a leg's last step still gets the
        # next Context (the driver instead stops exactly on the boundary so it
        # can record the closing stage's own rows first — see
        # driver.advance_preparation, which calls begin_next_leg itself).
        while not self.done and kernel.current_step >= self.leg_end:
            self.begin_next_leg()
        if self.done:
            # production is the DRIVER's leg (its runner owns the target,
            # reporting and checkpoint cadence): entering it ends preparation,
            # so a preparation step never runs there
            return kernel.current_step
        phase = self.phase
        end = self.leg_end
        interval = self.interval  # output.gamd.sample_interval
        cap = min(end, kernel.current_step + max_steps)
        reached_end = False
        while kernel.current_step < cap:
            # leg-relative lattice: every stage starts its sampling at 0
            next_sample = ((kernel.current_step // interval) + 1) * interval
            target = self.run.next_exchange_boundary(min(cap, next_sample))
            before = kernel.current_step
            kernel.step(target - before)
            self._stepped += target - before
            self.run.exchange_at_boundary()
            sample = phase in _SAMPLE_PHASES and (target == end or target == next_sample)
            if sample:
                readings = kernel.boost_potentials()
                values = {label: float(readings[label].energy) for label in self.run.channels}
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError("nonfinite GaMD calibration energy")
                record = {"step": target, "phase": phase, "energies": values}
                self.records.append(record)
                self._accumulate(record)
                # Parameters change ONLY at a stage boundary, never inside a
                # stage: the trajectory of every stage runs under the set it
                # started with, and the samples of a source stage calibrate
                # the next boost stage (see _handoff_source).
                if target == end:
                    self._calibrate_handoff(phase)
            if target == end:
                reached_end = True
        # The journal is cumulative, so rewriting it at every bounded turn is
        # unnecessary I/O (the driver now stops at each reporting-probe
        # boundary).  Flush on the checkpoint cadence the driver drives at,
        # plus every leg boundary; an explicit ``checkpoint()`` from the
        # driver (its configured cadence) still forces one.
        if reached_end or self._stepped - self._last_flush >= _FLUSH_INTERVAL:
            self.persist()
            self._last_flush = self._stepped
        return kernel.current_step


# ---------------------------------------------------------------------------
# the gamd.tsv reader + reweighting bridge (analysis stays the definition)
# ---------------------------------------------------------------------------


def read_gamd_trace(path) -> "tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]":
    """Parse ``gamd.tsv`` -> ``(steps, {label: dV}, {label: scale})``.

    Energies in kJ/mol; the step column is the first column (the GamdProbe
    layout — see :class:`neomd.probes.GamdProbe`).
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    rows = [line for line in lines
            if line.strip() and not line.startswith("#")]
    if not rows:
        raise ValueError(f"gamd trace {path!r} has no data rows")
    header = next(line for line in lines if line.startswith("#")).split("\t")
    labels = [column.rsplit("__", 1)[0]
              for column in header[1:] if column.endswith("__boost")]
    steps = np.asarray([int(row.split("\t")[0]) for row in rows],
                       dtype=np.int64)
    boost: dict[str, np.ndarray] = {}
    scale: dict[str, np.ndarray] = {}
    for label in labels:
        bcol = header.index(f"{label}__boost")
        scol = header.index(f"{label}__scale")
        boost[label] = np.asarray(
            [float(row.split("\t")[bcol]) for row in rows], dtype=np.float64)
        scale[label] = np.asarray(
            [float(row.split("\t")[scol]) for row in rows], dtype=np.float64)
    return steps, boost, scale


def reweight_observable(values, boost_kjmol, temperature: float):
    """Tiwary–Parrinello reweighting of one observable under GaMD boost.

    Thin bridge onto :func:`neomd.analysis.reweight.reweight_expectation`
    — the reweighting definition stays in the analysis subpackage.
    ``boost_kjmol`` is the ``<label>__boost`` column of gamd.tsv;
    unbiased expectations follow the weight w = exp(+β·ΔV).
    """
    from neomd.analysis.reweight import reweight_expectation

    return reweight_expectation(values, boost_kjmol, temperature)


# ---------------------------------------------------------------------------
# schema + registration (the knowledge triple)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": {
        "gamd": ("mapping: mode (total | dual), sigma0 (boost strength; absent "
                 "= 6 kcal/mol = 25.104 kJ/mol, a written number is in "
                 "energy_unit, default kJ/mol), "
                 "channels (optional explicit [{label, groups}] — LiGaMD "
                 "group definitions), calibration_input (optional external "
                 "energy tape / gamd_calibration.json), restart_from "
                 "(optional continue stage rerun); the sampling and recording "
                 "cadences live in output.gamd"),
        "steps": ("per-stage mapping {unbiased_eq, unbiased_production, "
                  "biased_eq, biased_eq2, production} — each value is that "
                  "stage's own step requirement and the length of its leg "
                  "(method 'gamd' only; ADR-0010 + ADR-0011); biased_eq2 "
                  "defaults to 0 and may be omitted"),
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "gamd.boosts": ("ordered [{label, selectors:[{atom_indices,force_types,include_boundary?}], "
                        "sigma0?, threshold_policy?, energy_unit?}] with optional final "
                        "{label,remaining:true}; force_types takes bond/angle/dihedral/"
                        "nonbonded or the scalar 'all'; include_boundary is true/false per "
                        "selected force type (bond/angle/dihedral default false, nonbonded "
                        "default true) and controls whether crossing terms join the boost; "
                        "a named type force_types does not select is an error; first-defined "
                        "owns each term; restraints excluded"),
        "gamd.threshold_policy": "upper (default, lower fallback) or lower, independently overridable per boost",
        "gamd.energy_unit": "kJ/mol (default) or kcal/mol; kcal/mol is explicitly multiplied by 4.184",
        "output.gamd.sample_interval": ("steps between the equilibration legs' V(P) "
                                        "statistics samples that calibrate the next boost "
                                        "stage (default 1000); the stage lengths must span "
                                        "at least a few samples"),
        "output.gamd.trace_interval": ("steps between production.tsv boost rows (default "
                                       "5000); rows a saved frame does not match add bytes, "
                                       "not information, so this is the fallback cadence "
                                       "when no finer frame source exists"),
        "output.gamd.report": "boolean, default true: write the production.tsv boost trace (the CE2 analysis input)",
        "gamd.calibration_input": ("optional path to an energy-sample tape (gamd.tsv / "
                                   "state.tsv — the latter feeds only the total channel) or a "
                                   "gamd_calibration.json whose (threshold, k) are adopted "
                                   "verbatim; the file's sha256 is recorded as provenance and "
                                   "the internal statistics stage never recomputes parameters"),
        "gamd.restart_from": ("optional stage name; with continue_md, rerun from the "
                              "previous stage's boundary checkpoint (ckpt/<stage>.ckpt) "
                              "instead of resuming mid-stage; parameters are pinned to "
                              "the boundary calibration unless calibration_input is given"),
        "gamd.exchange": ("experimental deterministic ligand relabel: copies, reference_atoms, interval? (default 1), "
                          "designated_slot?, ligand_reference_atoms? (copy offsets), cutoff_nm?, periodic?; "
                          "with cutoff retain designated copy at distance <= cutoff; "
                          "only when designated is outside swap the nearest other copy at distance <= cutoff; "
                          "without cutoff choose nearest of all slots; no energy or Metropolis acceptance gate; "
                          "not a qualified equilibrium PMF sampler"),
        "gamd.channels": ("legacy explicit channel definitions [{label, groups}] "
                          "over force-group ids (LiGaMD: point at the "
                          "ligand dihedral / ligand-nonbonded groups of a "
                          "system that separates them into their own "
                          "force groups)"),
        "continue_md": ("bool; restore output.ckpt (or input_files.checkpoint), trim "
                        "tapes, and continue — mid-stage resume in a gamd directory, "
                        "adopt from a foreign checkpoint, or with gamd.restart_from "
                        "a boundary rerun"),
        "output.gamd": ("mapping: report (bool, default true — the gamd.tsv "
                        "boost-trace switch the driver owns), sample_interval "
                        "(equilibration V(P) statistics cadence, default "
                        "1000), trace_interval (production tape cadence, "
                        "default 5000)"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'gamd'."""
    return GamdRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", LABEL, Method(schema=SCHEMA, prepare=_prepare))
