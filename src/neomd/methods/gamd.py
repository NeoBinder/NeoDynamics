"""Gaussian-accelerated MD (GaMD) — a method knowledge triple (schema +
prepare + the staged calibration/boost run).  See ADR-0005
(docs/adr/0005-gamd-boost-seam.md), ADR-0010 (per-stage steps input,
boundary restart and the AMBER adaptation schedule), ADR-0011 (each stage
is an independent LEG with its own Context and step axis) and
docs/methods/gamd.md.  Registers method: ``gamd``.
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

#: Sampled stages: every phase whose channel energies land in the journal
#: (diagnostics for the boosted equilibration, statistics for the two
#: AMBER statistics stages below).
_SAMPLE_PHASES = ("unbiased_production", "biased_eq", "biased_eq2")

#: Statistics stages — AMBER's staged protocol (GaMD_Amber-manual,
#: gamd-openmm stage_integrator): Vmax/Vmin/Vavg/sigmaV are collected in the
#: cMD statistics stage (``unbiased_production``) and updated FURTHER in the
#: adaptive bias collection stage (``biased_eq2``, AMBER's nteb); the bias
#: preparation stage (``biased_eq``, ntebprep) and production run under the
#: parameters fixed at their start.  Push schedule: once at the END of
#: ``unbiased_production``, then every ``steps.ntave``-step window inside
#: ``biased_eq2`` (plus its leg end) — parameters MOVE inside that one leg,
#: a deliberate ADR-0010 amendment; production never retunes.
_STATS_PHASES = ("unbiased_production", "biased_eq2")

#: Checkpoint-metadata key holding the leg the snapshot belongs to.  A leg's
#: clock is relative to its own Context and its RNG stream is that Context's,
#: so the index is state and must travel with the blob (ADR-0011): the journal
#: alone cannot disambiguate a clock of 0 between two adjacent leg entries.
_LEG_METADATA_KEY = "gamd_leg"

#: Checkpoint-metadata key carrying the per-step adaptation statistics
#: (Vmax/Vmin extremes, window + cumulative Welford, the open window) and
#: the live channel parameters, together with the leg/step they were fed
#: through.  The statistics are fed every step (AMBER/GaMD-OpenMM), so the
#: journal's lattice samples cannot rebuild them; the envelope is atomic
#: with the Context state, making a mid-leg resume of the ADAPTATION exact
#: (the trajectory itself stays exact only in NVT, ADR-0009).
_STATS_METADATA_KEY = "gamd_stats"

#: Steps between cumulative-journal flushes inside one preparation
#: call.  The driver now stops at every reporting-probe boundary, so
#: flushing per stop would rewrite an ever-growing journal O(n) times;
#: the driver's own checkpoint cadence still flushes explicitly.
_FLUSH_INTERVAL = 5000

#: The built-in boost-strength default: 6 kcal/mol (gamd-openmm's literature
#: value for the default whole-system boost).  It applies when ``gamd.sigma0`` is absent; a
#: number written in the YAML is interpreted in ``gamd.energy_unit`` (kJ/mol
#: unless that key says otherwise).  Everything internal is kJ/mol.
DEFAULT_SIGMA0_KCAL_MOL = 6.0


def default_ntave(num_particles: int) -> int:
    """The ``steps.ntave`` default: AMBER's ~4x-system-size recommendation.

    Amber 26 manual: "The default is set to 50,000 for GaMD simulations.  It
    is recommended to be updated as about 4 times of the total number of
    atoms in the system."  We implement the recommendation (not AMBER's flat
    50,000) snapped to the 1-2-5 significant-digit series: write 4N as
    a*10^n with 1 <= a < 10, pick whichever of 1, 2, 5, 10 is nearest to a
    (the 10 candidate covers rounding up into the next decade) and scale it
    back.  Ties pick the smaller value (deterministic).
    """
    a = float(4 * num_particles)
    n = 0
    while a >= 10.0:
        a /= 10.0
        n += 1
    pick = min((1.0, 2.0, 5.0, 10.0), key=lambda c: (abs(c - a), c))
    return max(1, int(pick * 10 ** n))


#: the ``gamd:`` plan section defaults (per-stage lengths live in the
#: plan-level ``steps`` mapping, see STAGE_NAMES)
DEFAULT_SETTINGS = {
    # The boost layout is one vocabulary: ``boosts`` (independent, possibly
    # overlapping claims; a final ``remaining: true`` boost takes the exact
    # complement).  A plan that writes no ``boosts`` is normalized by
    # ``plan._derive`` into the explicit default boost1 (whole system, every
    # force type, sigma0 = 6 kcal/mol, lower threshold policy), so this
    # default is only a fallback for direct construction.
    # None = the literature default, ``DEFAULT_SIGMA0_KCAL_MOL`` kcal/mol; a
    # value written in the YAML is a plain number in ``energy_unit``
    "sigma0": None,
    "boosts": None,  # ordered independent boost claims; never implicit remainder
    "threshold_policy": "upper",  # preserves historical upper-with-lower-fallback
    "energy_unit": "kJ/mol",
    "boost_cv_mode": "dv",  # dv | v0_dv: CV carries dV only, or V0 + dV
    "emit_system": None,  # {stage, E, K}: write the System XML, do not simulate
    "exchange": None,
    "calibration_input": None,  # external energy tape / gamd_calibration.json
    "restart_from": None,  # continue_md: rerun from this stage's boundary
}

#: pre-boosts layout keys that were removed; reported with migration
#: guidance instead of a bare unknown-key error
_REMOVED_KEYS = {
    "mode": (
        "the whole-system boost is the plan-level default boost1 (write no "
        "key); a dihedral channel is a boost with atom_indices: all + "
        "force_types: [dihedral]; AMBER's dual boost is two overlapping "
        "boosts — {force_types: all} plus {force_types: [dihedral]}"),
    "channels": (
        "define the partition with gamd.boosts selectors over atom groups "
        "and force types instead of force-group ids"),
}

#: The sampling/recording cadences are OUTPUT detail, so they live in the
#: method's own output namespace (``output.gamd``, ADR-0011) rather than in
#: the science section or in the analysis.  The defaults are deliberately
#: coarse: the CE2 analysis can only use rows a saved frame matches, so a
#: denser boost tape adds bytes, not information.  ``neomd.plan.gamd_output``
#: owns the defaults; this mirrors them for direct GamdRun construction.
#: (plan.gamd_output resolves the REAL defaults — including the
#: trace_interval sync with trajectory_interval; these are the bare-plan
#: fallback the plan-side reader fills in before this mirror ever applies)
OUTPUT_DEFAULTS = {"report": True, "sample_interval": 1000,
                   "trace_interval": 5000}

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
    plain ``state.tsv`` energy log (its ``potential`` column feeds only a
    SINGLE whole-system channel; per-channel energies need a gamd tape).  The file
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
        elif len(labels) == 1 and "potential" in header:
            index = header.index("potential")
        else:
            raise ValueError(
                f"calibration_input {path!r} lacks a {column!r} column for "
                f"boost channel {label!r} (a plain state.tsv can only feed a "
                "single whole-system boost)")
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
        removed = sorted(set(section) & set(_REMOVED_KEYS))
        if removed:
            raise ValueError(
                f"gamd section has removed keys {removed}: "
                + "; ".join(f"gamd.{key}: {_REMOVED_KEYS[key]}" for key in removed))
        unknown = set(section) - set(DEFAULT_SETTINGS)
        if unknown:
            raise ValueError(
                f"gamd section has unknown keys {sorted(unknown)} "
                f"(known: {sorted(DEFAULT_SETTINGS)})")
        settings.update(section)
        sigma0_written = "sigma0" in section
        sigma0 = float(settings["sigma0"]) if sigma0_written else None
        if sigma0 is not None and (not np.isfinite(sigma0) or sigma0 <= 0.0):
            raise ValueError(f"gamd.sigma0 must be > 0, got {sigma0}")
        self.settings = settings
        if settings["energy_unit"] not in ("kJ/mol", "kcal/mol"):
            raise ValueError("gamd.energy_unit must be kJ/mol or kcal/mol")
        self.energy_factor = 4.184 if settings["energy_unit"] == "kcal/mol" else 1.0
        if settings.get("boost_cv_mode", "dv") not in ("dv", "v0_dv"):
            raise ValueError("gamd.boost_cv_mode must be 'dv' or 'v0_dv'")
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
        # steps.ntave: the Vavg/sigmaV averaging window (AMBER's ntave).  An
        # explicit value wins; the default is AMBER's ~4x-system-size
        # recommendation snapped to the 1-2-5 series (default_ntave).
        ntave = getattr(plan, "gamd_ntave", None)
        self.ntave = (default_ntave(self.kernel.num_particles) if ntave is None
                      else int(ntave))
        # AMBER alignment: the two statistics stages must be whole ntave
        # windows ("ntcmdprep, ntcmd, ntebprep and nteb need to be multiples
        # of ntave") so every window the pushes read is complete.  Our
        # per-leg windows restart at leg entry, so ntcmdprep/ntebprep need
        # no constraint of their own; unbiased_production may be 0 when an
        # external calibration_input supplies the parameters (0 divides).
        for name in ("unbiased_production", "biased_eq2"):
            if self.gamd_steps[name] % self.ntave:
                raise ValueError(
                    f"steps.{name} must be a multiple of steps.ntave "
                    f"({self.ntave}): AMBER requires the statistics stages "
                    "to be whole ntave windows ('ntcmdprep, ntcmd, ntebprep "
                    "and nteb need to be multiples of ntave') so every "
                    "window the boost parameters read is complete - adjust "
                    f"steps.{name} or set steps.ntave explicitly")
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

        # -- boost claims: explicit boosts (plan normalization guarantees one)
        self.channels: dict[str, tuple[int, ...]] = {}
        if settings["boosts"]:
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
        else:
            # No explicit boosts: plan normalization synthesizes the default
            # whole-system boost1 (single source of truth, visible in the
            # journal/fingerprint).  Direct construction without that
            # normalization is a programming error, not a runtime default.
            raise ValueError(
                "gamd.boosts is required; the no-boosts default boost1 is "
                "synthesized during plan normalization")

        #: label -> the calibration outcome actually pushed to the kernel
        self.calibration: dict[str, dict] = {}

    # -- entry point --------------------------------------------------------

    def prepare(self):
        """Install zero-strength channels, plan the resume, build the stages.

        Channels are installed at ZERO strength — the same pre-Context
        discipline as ``install_bias`` — and the calibrated (threshold E,
        harmonic k) values are pushed through ``set_boost_param`` on
        AMBER's schedule (the BiasParamOps pattern applied to channels):
        once at the END of ``unbiased_production`` (stage 2), then every
        ``steps.ntave`` window inside ``biased_eq2`` (stage 4) plus its
        leg end; ``biased_eq`` (stage 3) and production run under the
        parameters fixed at their start — see ADR-0009/0010.  Plan
        normalization guarantees a nonempty ``boosts`` list (the no-boosts
        case becomes the explicit whole-system ``boost1``); the kernel's
        partition compiles the claims into per-channel groups (independent,
        possibly overlapping; a final ``remaining: true`` boost is the exact
        complement of the union).  ``gamd.emit_system`` short-circuits before
        resume/Context creation and returns a dry PreparedMethod (the driver
        writes ``<stage>_system.xml``).

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
            (BoostChannelIR(label=label, groups=groups)
             for label, groups in self.channels.items()))

        # emit_system: a dry preparation — build the System exactly as the
        # requested stage would, apply the user's (E, k) as CV global defaults
        # and let the driver write <stage>_system.xml.  No resume, no Context,
        # no integration.
        emit = self.settings.get("emit_system")
        if emit is not None:
            self._apply_emit_parameters(emit)
            if self.sink is not None and self.partition_report is not None:
                self.sink.write_bytes(
                    "gamd_partitions.json",
                    json.dumps(self.partition_report, indent=2).encode())
            return PreparedMethod(total_steps=0)

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

    def exchange_at_boundary(self) -> bool:
        if self.exchange is not None and self.kernel.current_step % self.exchange.interval == 0:
            return bool(self.exchange.on_step(self.kernel.current_step))
        return False

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

    def _apply_emit_parameters(self, emit: dict) -> None:
        """Apply ``emit_system.E``/``K`` as the CV's global default values.

        Values are kJ/mol (E) and 1/(kJ/mol) (K) by contract, independent of
        ``gamd.energy_unit`` (which only rescales the boost definitions'
        sigma0).  Missing labels default to 0; unknown labels are an error.
        """
        energies = emit.get("E") or {}
        harmonics = emit.get("K") or {}
        unknown = (set(energies) | set(harmonics)) - set(self.channels)
        if unknown:
            raise ValueError(
                f"emit_system.E/K name unknown boost labels {sorted(unknown)}; "
                f"known: {sorted(self.channels)}")
        for label in self.channels:
            self.kernel.set_boost_param(
                label, "threshold", float(energies.get(label, 0.0)))
            self.kernel.set_boost_param(
                label, "k", float(harmonics.get(label, 0.0)))

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
        def atoms_key(selector):
            atoms = selector.get("atom_indices")
            if atoms == "all":
                return "all"  # the whole system, however it is spelled
            return sorted(int(a) for a in (atoms or []))

        return {"boosts": [
            {"label": boost.get("label"),
             "remaining": bool(boost.get("remaining")),
             "selectors": [
                 {"atom_indices": atoms_key(selector),
                  "force_types": sorted(_force_type_list(selector.get("force_types"))),
                  "include_boundary": {str(k): bool(v) for k, v in
                                       sorted((selector.get("include_boundary") or {}).items())}}
                 for selector in (boost.get("selectors") or [])]}
            for boost in boosts]}
    # no boosts: the plan-level default boost1
    return {"default": "boost1"}


def _canonical_layout(layout):
    """Normalize a recorded layout across the removed legacy spellings.

    Pre-boosts journals recorded ``{"mode": "total"}``; that layout is the
    same as today's boost-less default.  ``mode: dual`` and force-group
    ``channels`` layouts have no boosts spelling, so they stay distinct and
    (correctly) fail the layout gate on continue.
    """
    if isinstance(layout, dict) and set(layout) == {"mode"} and layout.get("mode") == "total":
        return {"default": "total"}
    return layout


def _force_type_list(types):
    """A selector's force types as an iterable (the ``all`` alias stays literal)."""
    if types is None:
        return []
    if isinstance(types, str):
        return [types]
    return list(types)


def _recorded_payload(run, saved):
    """The configuration a run recorded, or None when the journal kept none.

    Only the current schema's ``fingerprint_payload`` is consulted: GaMD's
    artifact formats are still in development, so there is no legacy-journal
    compatibility layer (an old directory must be adopted through
    ``input_files.checkpoint``).
    """
    payload = saved.get("fingerprint_payload")
    if isinstance(payload, dict) and isinstance(payload.get("settings"), dict):
        return payload
    return None


def _recorded_layout(run, saved):
    """The recorded boost layout, or None when the journal did not keep it."""
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
    if _canonical_layout(recorded) == _canonical_layout(current):
        return ""
    return f"recorded {_short(recorded)} -> now {_short(current)}"


def _short(value) -> str:
    text = json.dumps(value, sort_keys=True, default=list)
    return text if len(text) <= 60 else text[:57] + "..."


def _fingerprint_drift(recorded, current) -> str:
    """Name the keys that differ between a journal and this run's config.

    Best effort: a journal without a payload returns "".  The comparison is
    per top-level group (``settings.*`` / ``output.*``), which is exactly what
    the fingerprint hash covers.
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

    Every leg transition persists a restart checkpoint naming the state
    that BEGINS a stage — ``restart/<stage>.ckpt`` (zero-length stages
    share their successor's).  Those are the restore points
    ``restart_from`` re-enters.  ``completed`` (journal, schema 4) holds each stage's already
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
        self.cv_v0_log: list[dict] = []  # DEBUG: CV-vs-native V0 verification tape
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
                    f"{boundary} restart checkpoint, which does not exist "
                    "(the stage before it never completed); available "
                    f"restart checkpoints: {self._available_boundaries(run) or 'none'}")
            # A stage-head restart needs no separate entry checkpoint (the plan
            # derives restart/<stage>.ckpt); an explicitly supplied
            # input_files.checkpoint is allowed only as a copy of that same
            # stage-head blob — a rolling/mid-stage checkpoint would silently
            # restore the wrong state.
            entry = getattr(run.plan, "checkpoint", None)
            if entry is not None:
                try:
                    entry_bytes = Path(entry).read_bytes()
                except OSError:
                    entry_bytes = None
                if entry_bytes is None:
                    raise ValueError(
                        f"gamd.restart_from {run.restart_from!r} needs its "
                        f"stage-head checkpoint {boundary!r}; "
                        f"input_files.checkpoint {entry!r} cannot be read")
                if entry_bytes != run.sink.read_bytes(boundary):
                    raise ValueError(
                        f"gamd.restart_from {run.restart_from!r} restarts from "
                        f"the stage-head checkpoint {boundary!r}; "
                        f"input_files.checkpoint must point at it (or a "
                        f"byte-identical copy), not {entry!r} — a rolling "
                        "checkpoint is only valid for a mid-stage continue_md "
                        "without restart_from")
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
        # -- AMBER adaptation statistics (Vmax/Vmin running since collection
        # began; Vavg/σV from closed windows of ``steps.ntave`` steps; fed
        # EVERY STEP of the statistics legs — AMBER's per-step imax/imin,
        # GaMD-OpenMM's per-step Welford).  The accumulators AND the live
        # channel parameters ride the checkpoint envelope (``gamd_stats``,
        # the same NEOMD wrapper ``gamd_leg`` uses), so a resume restores
        # them at exactly the position the restored Context re-runs from;
        # the journal's lattice records are diagnostics and the legacy
        # fallback, never the statistics source.
        self.ntave = max(1, int(run.ntave))
        #: the kernel's per-step V0 fast read (the GaMD statistics
        #: integrator's globals), when the kernel offers one; None on
        #: kernels without it (the fake) — boost_potentials is the fallback
        fast = getattr(run.kernel, "boost_step_energies", None)
        self._step_energy_read = fast if callable(fast) else None
        self.extremes: dict[str, dict[str, float]] = {}
        self.cumulative: dict[str, dict] = {}
        self.window_stats: dict[str, dict] = {}
        self._window: dict[str, dict] = {}
        self._stats_phase: str | None = None
        self._open_window = 0
        self._stats_leg: int | None = None
        self._stats_step = 0
        if not self._restore_stats_state(run):
            # legacy directory (a checkpoint from before the envelope) or a
            # fresh/adopted run: the journal's lattice samples are the only
            # statistics available — replay them, at lattice fidelity
            if self.records and self.mode in ("resume", "restart"):
                run.log.warning(
                    "gamd: the restored checkpoint carries no adaptation-"
                    "statistics envelope (gamd_stats); rebuilding from the "
                    "journal's %d lattice samples — per-step statistics are "
                    "not recoverable across this version boundary",
                    len(self.records))
            for i, record in enumerate(self.records):
                # a completed leg's last sample is its leg end (advance
                # always samples at ``target == end``); the resumed leg in
                # progress is not — its final window stays open for the
                # continuation
                later = self.records[i + 1] if i + 1 < len(self.records) else None
                phase = record["phase"]
                leg_end = ((later is None or later["phase"] != phase)
                           and self.completed.get(phase, 0) >= self.configured.get(phase, 0))
                self._feed_stats(record, live=False, leg_end=leg_end)
                if leg_end and phase in _STATS_PHASES:
                    self._push_adapted(phase, boundary=True, record=False)
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
        """Commit the running leg and the adaptation statistics into every
        subsequent checkpoint snapshot."""
        from neomd.kernel.port import CheckpointMetadataOps, provides

        kernel = self.run.kernel
        if not provides(kernel, CheckpointMetadataOps):
            return
        metadata = dict(kernel.checkpoint_metadata() or {})
        metadata[_LEG_METADATA_KEY] = int(self.index)
        snapshot = self._stats_snapshot()
        if snapshot is None:
            metadata.pop(_STATS_METADATA_KEY, None)
        else:
            metadata[_STATS_METADATA_KEY] = snapshot
        kernel.set_checkpoint_metadata(metadata)

    def _stats_snapshot(self):
        """The per-step statistics state + live parameters for the envelope.

        ``None`` while nothing has been fed (fresh runs before the first
        statistics step): the key stays out of the envelope so a restore of
        such a checkpoint starts statistics from scratch.
        """
        if self._stats_leg is None:
            return None
        return {
            "leg": int(self._stats_leg),
            "step": int(self._stats_step),
            "stats_phase": self._stats_phase,
            "open_window": int(self._open_window),
            "extremes": {label: {"min": float(w["min"]), "max": float(w["max"])}
                         for label, w in self.extremes.items()},
            "cumulative": {label: {"count": int(w["count"]), "mean": float(w["mean"]),
                                   "m2": float(w["m2"])}
                           for label, w in self.cumulative.items()},
            "window_stats": {label: {"count": int(w["count"]), "mean": float(w["mean"]),
                                     "m2": float(w["m2"])}
                             for label, w in self.window_stats.items()},
            "window": {label: {"count": int(w["count"]), "mean": float(w["mean"]),
                               "m2": float(w["m2"])}
                       for label, w in self._window.items()},
            "parameters": self.run.calibration,
        }

    def _restore_stats_state(self, run) -> bool:
        """Restore the per-step statistics from the checkpoint envelope.

        ``gamd_stats`` is atomic with the Context state inside the same
        checkpoint blob, so wherever it validates it is authoritative:

        * mid-leg resume: ``leg == self.index`` with ``step == completed``
          (the re-run feeds its tail on top of exactly this state);
        * a completed earlier leg: ``leg < self.index`` with ``step ==
          configured[leg]`` (any leg after it that also fed statistics
          would have moved the position itself).

        The envelope also carries the live ``(E, k)`` the interrupted
        Context ran with, which restores ``run.calibration`` — without it,
        a mid-adaptive-leg resume would restart from the previous stage's
        set, diverging from the uninterrupted run before its next window
        push.  ``False`` (absent or unvalidatable) falls back to the
        journal's lattice replay.
        """
        from neomd.kernel.port import CheckpointMetadataOps, provides

        if not provides(run.kernel, CheckpointMetadataOps):
            return False
        snap = (run.kernel.checkpoint_metadata() or {}).get(_STATS_METADATA_KEY)
        if not isinstance(snap, dict):
            return False
        try:
            leg = int(snap["leg"])
            step = int(snap["step"])
            if not 0 <= leg < len(ALL_STAGE_NAMES):
                return False
            if leg == self.index:
                if step != self.completed[ALL_STAGE_NAMES[leg]]:
                    return False
            elif leg > self.index or step != self.configured[ALL_STAGE_NAMES[leg]]:
                return False
            def welford(raw):
                return {label: {"count": int(w["count"]), "mean": float(w["mean"]),
                                "m2": float(w["m2"])}
                        for label, w in raw.items()}
            extremes = {label: {"min": float(w["min"]), "max": float(w["max"])}
                        for label, w in snap["extremes"].items()}
            cumulative = welford(snap["cumulative"])
            window_stats = welford(snap["window_stats"])
            window = welford(snap["window"])
            open_window = int(snap["open_window"])
            stats_phase = snap["stats_phase"]
            if stats_phase is not None and not isinstance(stats_phase, str):
                return False
            parameters = snap.get("parameters") or {}
            if not isinstance(parameters, dict):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        self.extremes = extremes
        self.cumulative = cumulative
        self.window_stats = window_stats
        self._window = window
        self._open_window = open_window
        self._stats_phase = stats_phase
        self._stats_leg, self._stats_step = leg, step
        if parameters:
            # the live set the interrupted context ran with; parameter
            # history and revisions stay owned by the journal
            run.calibration = parameters
        return True

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
        """Restart checkpoint holding the state that BEGINS stage ``index``."""
        return f"restart/{ALL_STAGE_NAMES[index]}.ckpt"

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
            # The CV energy-model mode changes the System composition itself
            # (dv keeps the native forces; v0_dv removes the covered ones and
            # puts their V0 in the CV), so a restored Context cannot switch
            # it: refuse before the layout comparison.
            payload = _recorded_payload(run, saved) or {}
            recorded_mode = (payload.get("settings") or {}).get("boost_cv_mode", "dv")
            if recorded_mode != run.settings.get("boost_cv_mode", "dv"):
                raise ValueError(
                    "GaMD boost_cv_mode changed since checkpoint "
                    f"({recorded_mode!r} -> {run.settings.get('boost_cv_mode')!r}); "
                    "the restored System has a different force composition — "
                    "adopt the run's output.ckpt into a fresh run to switch modes")
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
        def _history_kept(stage, entry):
            position = ALL_STAGE_NAMES.index(stage)
            if position < self.index:
                return True
            if position > self.index:
                return False
            # the current leg's entries survive only up to the restored
            # position: past it they describe pushes the tail re-run will
            # redo anyway (the journal may have flushed ahead of the
            # checkpoint), so keeping them would publish future parameters
            return int(entry.get("step", 0)) <= self.completed.get(stage, 0)

        run.parameter_history = {
            stage: entry for stage, entry in saved["parameter_history"].items()
            if _history_kept(stage, entry)}
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

    # -- CV-vs-native V0 verification tape (DEBUG, one-line opt-in) --------

    def record_cv_v0(self, step):
        """Record one frame's V0 cross-check (CV path vs native system).

        DEBUG/verification only — call this from :meth:`advance` (see the
        one-line call there, comment it out to disable).  Each frame appends
        rows to ``self.cv_v0_log`` (a plain list of dicts; cheap to append,
        tiny to hold) comparing, at the SAME configuration, the live Context
        against a Reference context over the pre-boost native System
        (``kernel.verify_cv_v0``):

        * ``v0_cv``    — the CV's value (``getCollectiveVariableValues``):
                         the bare boosted-region energy P, no boost.
        * ``dV``       — the boost potential this frame, from the CV's own
                         (threshold, k).
        * ``e_native`` — the native System's energy (no boost layer) at the
                         same positions/box.
        * ``e_total``  — the live Context's whole potential.
        * ``residual`` — ``e_total - e_native - Σ dV``: the invariant the
                         new design promises; ≈ 0 is the assertion, anything
                         else means the CV carries more than dV.

        Rows are flushed per stage to ``gamd_cv_v0/<stage>.tsv``.
        """
        if not self.cv_v0_log:
            self.cv_v0_log = []
        check = self.run.kernel.verify_cv_v0() if hasattr(self.run.kernel, "verify_cv_v0") else {}
        if not check:
            return
        for label, row in check.items():
            self.cv_v0_log.append({"step": int(step), "phase": str(self.phase),
                                   "label": label, "v0_cv": row["v0_cv"],
                                   "dV": row["dV"], "e_total": row["e_total"],
                                   "e_native": row["e_native"],
                                   "threshold": row["threshold"], "k": row["k"]})

    def flush_cv_v0(self):
        """Write each stage's verification rows to its OWN ``gamd_cv_v0/<stage>.tsv``.

        One file per stage (like the per-stage rolling ``<stage>.ckpt``): the rows recorded
        during ``unbiased_eq`` land in ``gamd_cv_v0/unbiased_eq.tsv``,
        ``unbiased_production`` in its own file, and so on.  Each file is
        rewritten (atomic replace) with only that stage's rows, so reading a
        file is exactly "what this stage saw".
        """
        if not getattr(self, "cv_v0_log", None) or self.run.sink is None:
            return
        header = "step\tphase\tlabel\tv0_cv\tdV\te_total\te_native\tthreshold\tk"
        # group rows by the phase they were recorded in (one file per stage)
        by_phase: dict[str, list[dict]] = {}
        for row in self.cv_v0_log:
            by_phase.setdefault(row.get("phase", "unknown"), []).append(row)
        for phase, rows in by_phase.items():
            lines = [header] + [
                f"{r['step']}\t{r['phase']}\t{r['label']}\t{r['v0_cv']:.6f}"
                f"\t{r['dV']:.6f}\t{r['e_total']:.6f}\t{r['e_native']:.6f}"
                f"\t{r['threshold']:.6f}\t{r['k']:.6f}"
                for r in rows]
            self.run.sink.write_bytes(
                f"gamd_cv_v0/{phase}.tsv", ("\n".join(lines) + "\n").encode())

    def _accumulate(self, record):
        moments = self.moments[record["phase"]]
        for label, value in record["energies"].items():
            m = moments.setdefault(label, {"count": 0, "mean": 0., "m2": 0., "min": value, "max": value})
            m["count"] += 1
            delta = value - m["mean"]
            m["mean"] += delta / m["count"]
            m["m2"] += delta * (value - m["mean"])
            m["min"], m["max"] = min(m["min"], value), max(m["max"], value)

    # -- AMBER adaptation statistics and parameter pushes --------------------

    def _feed_stats(self, record, live, leg_end=False):
        """Feed one step's channel energies into the adaptation statistics.

        Vmax/Vmin run since collection BEGAN (they carry from the cMD stage
        into the adaptive stage — AMBER's continuation); Vavg/σV come from
        closed windows of ``steps.ntave`` steps.  Live feeding happens
        EVERY step of a statistics leg (AMBER's per-step imax/imin,
        GaMD-OpenMM's per-step Welford — ``advance`` steps one step at a
        time there to read each step's P), so a window of ``ntave`` steps
        closes exactly on its boundary step, and a leg's last step seals
        its final (possibly partial) window.  ``live=False`` is the legacy
        replay of a journal's lattice samples: sparser feeding with the
        same window arithmetic (the first sample beyond a boundary closes
        it), re-applying parameters without recording revisions — the
        journal already carries them, and a tail rerun re-records what its
        filter dropped.
        """
        phase = record["phase"]
        if phase not in _STATS_PHASES:
            return
        if phase != self._stats_phase:
            # a new statistics leg starts a FRESH window (GaMD-OpenMM resets
            # windowCount at stage boundaries); the extremes carry over
            self._stats_phase = phase
            self._window = {}
            self._open_window = 0
        step = int(record["step"])
        for label, value in record["energies"].items():
            extreme = self.extremes.setdefault(label, {"min": value, "max": value})
            extreme["min"] = min(extreme["min"], value)
            extreme["max"] = max(extreme["max"], value)
            for store in (self.cumulative, self._window):
                w = store.setdefault(label, {"count": 0, "mean": 0., "m2": 0.})
                w["count"] += 1
                delta = value - w["mean"]
                w["mean"] += delta / w["count"]
                w["m2"] += delta * (value - w["mean"])
        # the position the statistics were fed through (envelope validation
        # on resume): every feeding call moves it, live or replay
        self._stats_leg = ALL_STAGE_NAMES.index(phase)
        self._stats_step = step
        window_index = step // self.ntave
        if leg_end or window_index > self._open_window:
            self._close_window()
            self._open_window = window_index
            if phase == "biased_eq2" and not leg_end:
                # the in-leg adaptation push: parameters MOVE inside this
                # one leg (AMBER's nteb; ADR-0010 amendment)
                self._push_adapted(phase, record=live)

    def _close_window(self):
        """Seal the open window's Welford state as the latest Vavg/σV."""
        if any(w["count"] for w in self._window.values()):
            self.window_stats = {label: dict(w) for label, w in self._window.items()}
        self._window = {}

    def _adaptation_moments(self):
        """(min, max, Vavg, σV², count) per channel for the next push.

        Vmax/Vmin are the running extremes since collection began; Vavg/σV
        come from the last closed window (when it holds a spread), else the
        whole collection.  A single-sample stage still yields a row — its
        degenerate statistics select ZERO strength, which the boundary push
        records exactly like the pre-window code did.
        """
        out = {}
        for label, extreme in self.extremes.items():
            window = self.window_stats.get(label)
            cumulative = self.cumulative.get(label)
            for stats in (window, cumulative):
                if stats is not None and stats["count"] >= 2:
                    out[label] = {"min": extreme["min"], "max": extreme["max"],
                                  "mean": stats["mean"], "m2": stats["m2"],
                                  "count": stats["count"]}
                    break
            else:
                stats = window or cumulative
                if stats is not None:
                    out[label] = {"min": extreme["min"], "max": extreme["max"],
                                  "mean": stats["mean"], "m2": stats["m2"],
                                  "count": stats["count"]}
        return out

    def _parameters_from(self, moments):
        """(E, k) per channel from one statistics mapping (the Calc_E_k0)."""
        calibration = {}
        for label, moment in moments.items():
            settings = self.run.channel_settings.get(label, {})
            sigma0 = settings.get("sigma0", self.run.sigma0)
            policy = settings.get("threshold_policy", self.run.settings["threshold_policy"])
            calibration[label] = _params_from_statistics(
                moment["min"], moment["max"], moment["mean"],
                float(np.sqrt(max(0., moment["m2"] / moment["count"]))),
                sigma0, policy)
        return calibration

    def _push_adapted(self, phase, *, boundary=False, record=True):
        """Recompute (E, k) from the adaptation statistics and push them.

        The schedule is AMBER's (user-confirmed): stage 2 pushes once at its
        END (its set starts every boost leg); stage 4 (``biased_eq2``) pushes
        at every ntave window it closes plus its leg end, so production
        freezes the freshest set; stage 3 (``biased_eq``) and production
        never push.  An external ``gamd.calibration_input`` pins every
        channel, so statistics are recorded but never recomputed.

        A stage-2 push with no samples at all keeps the installed set and
        warns loudly (a production run under zero-strength boosts is a
        scientific no-op); any push from a BOOSTED source (in-leg window or
        the adaptive stage's end) whose statistics are too flat (fewer than
        two samples, or a channel the selection would disable) keeps the
        incoming set instead of disabling the channel outright.
        """
        if self.run.calibration_input is not None:
            return
        moments = self._adaptation_moments()
        if not moments:
            if boundary and phase == "unbiased_production":
                # NOT silent: with per-step feeding, empty statistics mean
                # the stage never ran a step, so the boost legs keep
                # whatever (possibly zero) strength is installed.
                self.run.log.warning(
                    "gamd %s produced no V(P) statistics steps: the boost "
                    "stages keep the currently installed boost parameters, "
                    "which means ZERO strength on a fresh run; lengthen "
                    "the stage", phase)
            else:
                self.run.log.info(
                    "gamd %s produced no V(P) statistics steps; keeping the "
                    "installed boost parameters", phase)
            return
        if not boundary or phase == "biased_eq2":
            # a boosted source (in-leg window or the adaptive stage's end)
            # with a single sample or no usable spread keeps the incoming
            # set instead of disabling the channel outright; the un-boosted
            # statistics stage pushes unconditionally (fresh runs have no
            # incoming set to keep)
            fit = self._parameters_from(moments)
            if (min(moment["count"] for moment in moments.values()) < 2
                    or any(params["disabled_reason"] for params in fit.values())):
                self.run.log.info(
                    "gamd %s statistics too flat to re-tune; keeping the "
                    "current boost parameters", phase)
                return
        if record:
            self.run.complete_calibration(
                {label: () for label in moments}, moments=moments, persist=False,
                stage=phase)
        else:
            # replay: identical values over identical samples — re-apply
            # without a new revision (the journal owns the history)
            self.run.calibration = self._parameters_from(moments)
            self.run._push_calibration()

    def _final_source(self, configured):
        """The stage whose statistics define the last boost stage's set."""
        return "biased_eq2" if configured.get("biased_eq2", 0) > 0 else "unbiased_production"

    def reconcile_handoff(self):
        """Keep production's parameters tied to its last boost equilibration.

        An unchanged configuration needs nothing: the journal's last entry
        already is the set the last boost stage ended with.  Only when the
        saved layout put a *different* stage at the end of the chain (e.g. a
        resume that adds or drops ``biased_eq2``) is the set re-derived from
        the new source stage's restored statistics, so production cannot
        inherit parameters from a stage layout that no longer exists.
        """
        saved = self._saved_configured
        if saved is None or self.run.calibration_input is not None:
            return
        source = self._final_source(self.configured)
        if source == self._final_source(saved):
            return
        if self._completed_now()[source] < self.configured[source]:
            return  # still to run: advance() pushes at its windows/end
        moments = self._adaptation_moments() if self.extremes else {}
        if not moments:
            moments = self.moments.get(source) or {}
        if not moments:
            return
        calibration = self._parameters_from(moments)
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
            # self.flush_cv_v0()  # DEBUG: write gamd_cv_v0.tsv (comment out to disable)
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

        Statistics legs (``unbiased_production``/``biased_eq2``) integrate
        ONE STEP AT A TIME: every step's channel energies feed Vmax/Vmin
        and the window Welford immediately (AMBER's per-step imax/imin,
        GaMD-OpenMM's per-step Welford — their CustomIntegrator globals,
        read method-side because our P lives inside the CustomCVForce where
        an integrator expression cannot see it).  Non-statistics legs keep
        chunked stepping: nothing is read between the journal's lattice
        points there.
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
        per_step = phase in _STATS_PHASES
        while kernel.current_step < cap:
            if per_step:
                # step(1) at a time: the next step's P is only readable
                # through the step's own force evaluation, so intermediate
                # steps cannot be skipped (exchange boundaries are still
                # hit exactly — every step is a loop boundary now)
                target = kernel.current_step + 1
            else:
                # leg-relative lattice: every stage starts its sampling at 0
                next_sample = ((kernel.current_step // interval) + 1) * interval
                target = self.run.next_exchange_boundary(min(cap, next_sample))
            before = kernel.current_step
            kernel.step(target - before)
            self._stepped += target - before
            exchanged = self.run.exchange_at_boundary()
            if per_step:
                self._read_step(phase, target, end, interval,
                                fresh=exchanged)
            else:
                self._record_lattice(phase, target, end, target == next_sample)
            # self.record_cv_v0(target)  # DEBUG: CV-vs-native V0 check (comment out to disable)
            if target == end:
                reached_end = True
        # the statistics state moved with every step above: commit it so the
        # driver's next checkpoint envelope carries the exact position
        self._sync_checkpoint_metadata()
        # The journal is cumulative, so rewriting it at every bounded turn is
        # unnecessary I/O (the driver now stops at each reporting-probe
        # boundary).  Flush on the checkpoint cadence the driver drives at,
        # plus every leg boundary; an explicit ``checkpoint()`` from the
        # driver (its configured cadence) still forces one.
        if reached_end or self._stepped - self._last_flush >= _FLUSH_INTERVAL:
            self.persist()
            self._last_flush = self._stepped
        return kernel.current_step

    def _read_step(self, phase, step, end, interval, *, fresh=False):
        """Read one step's channel energies and feed them where they go.

        The adaptation statistics feed on EVERY step (that is the whole
        point of the per-step loop); the journal's diagnostic record lands
        on the ``sample_interval`` lattice (and always on the leg's last
        step).  ``leg_end`` seals the final (possibly partial) window and
        the boundary push follows it — stage 2 hands its set to the boost
        legs, stage 4 leaves production the freshest set; biased_eq2 also
        pushed at every window it closed inside the leg.

        Group-readable channels use the CustomIntegrator's per-step V0
        global (a few microseconds).  Clone/CV channels have no fast value
        and are read fresh through ``boost_readings``.  ``fresh=True`` is
        used after a ligand exchange actually moved coordinates: every
        channel must then come from the post-exchange configuration, so the
        pre-exchange integrator globals are skipped.
        """
        kernel = self.run.kernel
        values: dict[str, float] = {}
        if not fresh:
            fast = self._step_energy_read
            if fast is not None:
                values.update({label: float(value)
                               for label, value in fast().items()})
        missing = [label for label in self.run.channels if label not in values]
        if missing:
            reader = getattr(kernel, "boost_readings", None)
            if not callable(reader):
                reader = kernel.boost_potentials
            readings = reader()
            for label in missing:
                values[label] = float(readings[label].energy)
        bad = {label: value for label, value in values.items()
               if not np.isfinite(value)}
        if bad:
            # This check can fire BEFORE the platform's own NaN check
            # (an energy reads inf while the coordinates are still
            # finite), so the offending numbers themselves are the
            # diagnosis: name them, and let an attached watchdog dump
            # the window it kept.
            error = ValueError(
                f"nonfinite GaMD calibration energy: {bad} "
                "(channel energies at this step; set plan debug: true, or "
                "legacy NEOMD_WATCH=1, for the per-tile readings)")
            notifier = getattr(kernel, "notify_watch_failure", None)
            if callable(notifier):
                notifier(step, error)
            raise error
        leg_end = step == end
        record = {"step": step, "phase": phase, "energies": values}
        if leg_end or step % interval == 0:
            self.records.append(record)
            self._accumulate(record)
        # every step feeds the statistics — the record above is diagnostics
        self._feed_stats(record, live=True, leg_end=leg_end)
        if leg_end:
            self._push_adapted(phase, boundary=True)

    def _record_lattice(self, phase, target, end, on_lattice):
        """Non-statistics legs: journal + diagnostics at the lattice only."""
        if not (phase in _SAMPLE_PHASES and (target == end or on_lattice)):
            return
        readings = self.run.kernel.boost_potentials()
        values = {label: float(readings[label].energy) for label in self.run.channels}
        bad = {label: value for label, value in values.items()
               if not np.isfinite(value)}
        if bad:
            error = ValueError(
                f"nonfinite GaMD calibration energy: {bad} "
                "(channel energies at this sample step; set plan debug: "
                "true, or legacy NEOMD_WATCH=1, for the per-tile readings)")
            notifier = getattr(self.run.kernel, "notify_watch_failure", None)
            if callable(notifier):
                notifier(target, error)
            raise error
        record = {"step": target, "phase": phase, "energies": values}
        self.records.append(record)
        self._accumulate(record)


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
        "gamd": ("mapping: boosts (optional ordered boost claims; absent = plan-"
                 "level default boost1: atom_indices all + bond/angle/dihedral/"
                 "nonbonded, threshold_policy lower, sigma0 6 kcal/mol — AMBER's "
                 "igamd=1), sigma0 (boost strength; absent = 6 kcal/mol = "
                 "25.104 kJ/mol, a written number is in energy_unit, default "
                 "kJ/mol), calibration_input (optional external "
                 "energy tape / gamd_calibration.json), restart_from "
                 "(optional continue stage rerun), boost_cv_mode (dv | v0_dv), "
                 "emit_system (dry System-XML preparation, no simulation); the "
                 "sampling and recording cadences live in output.gamd"),
        "steps": ("per-stage mapping {unbiased_eq, unbiased_production, "
                  "biased_eq, biased_eq2, production} — each value is that "
                  "stage's own step requirement and the length of its leg "
                  "(method 'gamd' only; ADR-0010 + ADR-0011); ALL five stage "
                  "keys are required and each must be >= 1 (the strict "
                  "five-stage protocol: plain eq, cMD statistics, boost eq, "
                  "adaptive boost eq, boost production; the only exception "
                  "is unbiased_production: 0 with gamd.calibration_input); "
                  "the optional integer "
                  "ntave is the Vavg/sigmaV averaging window in steps "
                  "(AMBER's ntave; default: AMBER's ~4x-system-size "
                  "recommendation snapped to the 1-2-5 series — "
                  "neomd.methods.gamd.default_ntave): unbiased_production "
                  "and biased_eq2 must be whole multiples of ntave (AMBER: "
                  "'ntcmdprep, ntcmd, ntebprep and nteb need to be multiples "
                  "of ntave') so every window the pushes read is complete; "
                  "the statistics legs feed Vmax/Vmin and the window Welford "
                  "on EVERY step, and parameters are pushed at the end of "
                  "unbiased_production and at every ntave window inside "
                  "biased_eq2"),
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "gamd.boosts": ("ordered [{label, selectors:[{atom_indices,force_types,include_boundary?}], "
                        "sigma0?, threshold_policy?, energy_unit?}] with optional final "
                        "{label,remaining:true}; atom_indices is a nonempty list of "
                        "zero-based indices or the whole-system alias 'all' (AMBER's "
                        "igamd=1..5 semantics, e.g. every dihedral); force_types takes "
                        "bond/angle/dihedral/"
                        "nonbonded or the scalar 'all'; include_boundary is true/false per "
                        "selected force type (bond/angle/dihedral default false, nonbonded "
                        "default true) and controls whether crossing terms join the boost; "
                        "a named type force_types does not select is an error; claims are "
                        "INDEPENDENT — every boost owns everything its selectors match, so "
                        "overlapping boosts share terms (AMBER's dual boost: a "
                        "{force_types: all} boost plus a {force_types: [dihedral]} boost); "
                        "the final remaining:true boost is the exact complement of the "
                        "union of the others' coverage; one include_boundary.nonbonded "
                        "value per boost, and include=false sets within one boost must be "
                        "disjoint (their masked blocks M(S) would double-count); "
                        "restraints excluded; a channel owning whole forces "
                        "gets those natives moved into its pure force group and its V0 is "
                        "read as that group's energy (partial or shared selections read "
                        "the CV's clone values — identical numbers)"),
        "gamd.threshold_policy": "upper (default, lower fallback) or lower, independently overridable per boost",
        "gamd.boost_cv_mode": ("dv (default: V0 stays native; the CustomCVForce "
                               "carries only dV) or v0_dv (the whole native force "
                               "V0 plus dV live in the CV; covered native forces "
                               "are removed from the System, so every boosted "
                               "Force must be covered exactly once by the boost "
                               "channels — Σ channel V0 == native energy — else "
                               "preparation raises and asks for dv). Fixed for the "
                               "whole run and part of the journal fingerprint"),
        "gamd.emit_system": ("mapping {stage?, E?, K?}: a dry preparation that "
                             "builds the requested stage's System (same CV mode) "
                             "and writes <output_dir>/<stage>_system.xml, then "
                             "stops without creating a Context or integrating; "
                             "stage defaults to production; E is a {boost label: "
                             "threshold in kJ/mol} mapping and K a {boost label: "
                             "k in 1/(kJ/mol)} mapping, both defaulting to 0; "
                             "cannot be combined with continue_md"),
        "gamd.energy_unit": "kJ/mol (default) or kcal/mol; kcal/mol is explicitly multiplied by 4.184",
        "output.gamd.sample_interval": ("steps between the equilibration legs' "
                                        "DIAGNOSTIC V(P) samples journaled to "
                                        "gamd_stages.json (default 1000); the "
                                        "adaptation statistics themselves feed on "
                                        "EVERY step of the statistics legs (AMBER's "
                                        "per-step imax/imin, GaMD-OpenMM's per-step "
                                        "Welford — the journal's lattice is not the "
                                        "statistics source)"),
        "output.gamd.trace_interval": ("steps between production.tsv boost rows (default "
                                       "5000); rows a saved frame does not match add bytes, "
                                       "not information, so this is the fallback cadence "
                                       "when no finer frame source exists"),
        "output.gamd.report": "boolean, default true: write the production.tsv boost trace (the CE2 analysis input)",
        "gamd.calibration_input": ("optional path to an energy-sample tape (gamd.tsv / "
                                   "state.tsv — the latter feeds a single whole-system boost) or a "
                                   "gamd_calibration.json whose (threshold, k) are adopted "
                                   "verbatim; the file's sha256 is recorded as provenance and "
                                   "the internal statistics stage never recomputes parameters"),
        "gamd.restart_from": ("optional stage name; with continue_md, rerun from the "
                              "stage's restart checkpoint (restart/<stage>.ckpt) instead "
                              "of resuming mid-stage — no input_files.checkpoint is "
                              "needed (an explicit one must be a byte-identical copy of "
                              "that stage-head blob); parameters are pinned to the "
                              "boundary calibration unless calibration_input is given"),
        "gamd.exchange": ("experimental deterministic ligand relabel: copies, reference_atoms, interval? (default 1), "
                          "designated_slot?, ligand_reference_atoms? (copy offsets), cutoff_nm?, periodic?; "
                          "with cutoff retain designated copy at distance <= cutoff; "
                          "only when designated is outside swap the nearest other copy at distance <= cutoff; "
                          "without cutoff choose nearest of all slots; no energy or Metropolis acceptance gate; "
                          "not a qualified equilibrium PMF sampler"),
        "continue_md": ("bool; resume from the plan's resolved checkpoint — with "
                        "gamd.restart_from a stage-head restart (entry = "
                        "restart/<stage>.ckpt, no input_files.checkpoint needed), "
                        "otherwise a mid-stage resume that must name "
                        "input_files.checkpoint (adopting a foreign checkpoint is "
                        "the mid-stage form)"),
        "output.gamd": ("mapping: report (bool, default true — the gamd.tsv "
                        "boost-trace switch the driver owns), sample_interval "
                        "(equilibration V(P) statistics cadence, default "
                        "1000), trace_interval (production tape cadence; "
                        "defaults to the trajectory_interval cadence so "
                        "every saved DCD frame gets a CE2 row — flat 5000 "
                        "only with no trajectory)"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'gamd'."""
    return GamdRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", LABEL, Method(schema=SCHEMA, prepare=_prepare))
