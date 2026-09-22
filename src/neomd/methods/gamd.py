"""Gaussian-accelerated MD (GaMD) — a method knowledge triple (schema +
prepare + the calibration/boost run).  See ADR-0005
(docs/adr/0005-gamd-boost-seam.md) and docs/methods/gamd.md.
Registers method: ``gamd``.  Never imports openmm (methods stay
kernel-agnostic; the port's kJ/mol unit conventions throughout).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd.kernel.port import (
    BoostChannelIR,
    BoostOps,
    CheckpointMetadataOps,
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
    "read_gamd_trace",
    "reweight_observable",
]

LABEL = "gamd"
TAPE_FILENAME = "gamd.tsv"
CALIBRATION_FILENAME = "gamd_calibration.json"

LOG = logging.getLogger("neomd.methods.gamd")

#: the ``gamd:`` plan section defaults (σ0 in kJ/mol — the GaMD literature
#: default 6.0 kJ/mol ≈ 1.4 kcal/mol for both the total and dihedral
#: channels in gamd-openmm; steps land inside plan.steps, see the module
#: docstring)
DEFAULT_SETTINGS = {
    "mode": "total",  # "total" | "dual"  (explicit channels: see below)
    "sigma0": 6.0,  # kJ/mol — the threshold standard deviation
    "calibration_steps": 200,  # cMD pre-run length (steps)
    "calibration_interval": 10,  # steps between calibration samples
    "frequency": 10,  # steps between gamd.tsv rows
    "channels": None,  # optional [{label, groups}] explicit channel defs
    "boosts": None,  # ordered atom/type partitions; never implicit remainder
    "threshold_policy": "upper",  # preserves historical upper-with-lower-fallback
    "energy_unit": "kJ/mol",
    "preparation_steps": 0,
    "boost_preparation_steps": 0,
    "equilibration_steps": 0,
    "adaptation_interval": None,
    "exchange": None,
}

_VALID_MODES = ("total", "dual")


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method")."""

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one GaMD run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count (calibration + production)
    channels: dict  # label -> {threshold, k} actually pushed to the kernel
    mean_boost: dict  # label -> mean ΔV (kJ/mol) over the production rows
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# the calibration math — pure functions (one definition point, test-parseable)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class GamdRun:
    """One GaMD execution over a kernel (install → calibrate → boost).

    The calibration pre-run advances the kernel's step counter:
    ``plan.steps`` is the FINAL step of the whole run (calibration +
    boosted production), the same absolute-step convention ``resume`` uses.

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
        sigma0 = float(settings["sigma0"])
        if not np.isfinite(sigma0) or sigma0 <= 0.0:
            raise ValueError(f"gamd.sigma0 must be > 0, got {sigma0}")
        self.settings = settings
        if settings["energy_unit"] not in ("kJ/mol", "kcal/mol"):
            raise ValueError("gamd.energy_unit must be kJ/mol or kcal/mol")
        self.energy_factor = 4.184 if settings["energy_unit"] == "kcal/mol" else 1.0
        self.sigma0 = sigma0 * self.energy_factor
        if settings["threshold_policy"] not in ("lower", "upper"):
            raise ValueError("gamd.threshold_policy must be lower or upper")
        self.staged = bool(settings["boosts"]) or any(
            key in section for key in ("preparation_steps", "boost_preparation_steps",
                                       "equilibration_steps", "adaptation_interval"))
        self.stage = None
        self.exchange = None
        self.parameter_history = []
        self.control_history = []
        self.control_revision = 0
        self.channel_settings = {}
        self.partition_report = None
        self.calibration_steps = int(settings["calibration_steps"])
        self.calibration_interval = int(settings["calibration_interval"])
        if self.calibration_steps < 1:
            raise ValueError("gamd.calibration_steps must be >= 1")
        if self.calibration_interval < 1:
            raise ValueError("gamd.calibration_interval must be >= 1")
        self.frequency = int(settings["frequency"])
        if self.frequency < 1:
            raise ValueError(f"gamd.frequency must be >= 1, got {self.frequency}")

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
        """Install zero-strength channels, plan the resume, calibrate.

        Channels are installed at ZERO strength — the same pre-Context
        discipline as ``install_bias`` — and the calibrated (threshold E,
        harmonic k) values are pushed live afterwards (the BiasParamOps
        pattern applied to channels, via ``set_boost_param``).  ``mode:
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
                f"(install_boost / set_boost_param / boost_potentials); "
                f"GaMD cannot run on it")

        # zero strength: k = 0 (and a threshold far above any energy) —
        # the same dynamics as an un-boosted run (ADR-0005: calibration's
        # cMD phase has no bit-string-identity obligations)
        self.kernel.install_boost(
            BoostChannelIR(label=label, groups=groups)
            for label, groups in self.channels.items())

        # resume through the single owner — AFTER install_boost (the
        # openmm Context must come up with the boost integrator in place)
        from neomd.resume import plan_resume

        resume_plan = plan_resume(self.plan, self.kernel, self.sink)
        resumed = resume_plan is not None
        if provides(self.kernel, CheckpointMetadataOps):
            saved_control = self.kernel.checkpoint_metadata().get("gamd_control") if resumed else None
            if saved_control is not None:
                revision, sigma = saved_control["revision"], saved_control["sigma0"]
                if (type(revision) is not int or revision < 0
                        or not np.isfinite(sigma) or sigma <= 0):
                    raise ValueError("invalid GaMD calibration control checkpoint")
                self.control_revision, self.sigma0 = revision, float(sigma)
                for entry in self.settings["boosts"] or ():
                    if "sigma0" not in entry:
                        self.channel_settings[entry["label"]]["sigma0"] = self.sigma0
            self._checkpoint_control()
        if self.staged:
            self.stage = GamdStages(self, resumed=resumed)
        elif resumed:
            try:
                self._load_calibration()
            except (FileNotFoundError, KeyError):
                partial = None if self.sink is None or not self.sink.exists("gamd_preparation.json") else json.loads(
                    self.sink.read_bytes("gamd_preparation.json"))
                if partial is None or self.kernel.current_step >= partial["calibration_end"]:
                    raise FileNotFoundError(f"cannot continue gamd: {CALIBRATION_FILENAME} not found") from None
        if self.calibration:
            # fresh-with-saved-calibration or resume: push, never re-run
            # (pushing over a checkpoint is idempotent — ADR-0005)
            self._push_calibration()
        else:
            pass  # bounded calibration is advanced by the driver/session

        if self.settings["exchange"] is not None:
            from neomd.methods.ligand_exchange import LigandExchange
            self.exchange = LigandExchange(self.kernel, self.settings["exchange"], self.sink,
                                           resumed=resumed)
        if self.sink is not None:
            if self.partition_report is not None:
                self.sink.write_bytes("gamd_partitions.json", json.dumps(self.partition_report, indent=2).encode())
            from neomd.driver import timestep_ps
            self.sink.write_bytes("gamd_metadata.json", json.dumps({
                "schema_version": 1, "energy_unit": "kJ/mol", "time_unit": "ps",
                "labels": list(self.channels), "frame_semantics": "saved_configuration",
                "timestep_ps": timestep_ps(self.plan), "settings": self.settings,
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
                interval=self.frequency,
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
            preparation=self.stage if self.staged else (None if self.calibration else GamdCalibration(self)),
            before_step=None if self.exchange is None else self.exchange.on_step,
            before_step_interval=1 if self.exchange is None else self.exchange.interval,
            checkpoint=self.checkpoint,
            controls=({"sigma0": {"unit": "kJ/mol", "exclusive_minimum": 0.0, "maximum": 10000.0,
                                  "phase": "calibration"}}
                      if provides(self.kernel, CheckpointMetadataOps) else {}),
            update_parameter=self.update_parameter,
        )

    def metadata(self):
        return {"phase": self.stage.phase if self.stage is not None else ("production" if self.calibration else "statistics"),
                "parameter_revision": len(self.parameter_history),
                "slot_to_identity": [] if self.exchange is None else list(self.exchange.slot_to_identity)}

    def checkpoint(self):
        if self.stage is not None:
            self.stage.persist()
        if self.exchange is not None:
            self.exchange.persist()

    def update_parameter(self, name, value):
        """Calibration target may change before production parameters are set."""
        if name != "sigma0" or self.calibration:
            raise ValueError("sigma0 is live-settable only during GaMD calibration")
        if not provides(self.kernel, CheckpointMetadataOps):
            raise NotImplementedError("live GaMD calibration controls require checkpoint metadata support")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("sigma0 must be finite and positive (kJ/mol)")
        self.sigma0 = value
        self.control_revision += 1
        self.control_history.append({"step": int(self.kernel.current_step), "sigma0": value,
                                     "revision": self.control_revision})
        self._checkpoint_control()
        for entry in self.settings["boosts"] or ():
            if "sigma0" not in entry:
                self.channel_settings[entry["label"]]["sigma0"] = value
        self.checkpoint()

    def _checkpoint_control(self):
        metadata = self.kernel.checkpoint_metadata()
        metadata["gamd_control"] = {"revision": self.control_revision, "sigma0": self.sigma0}
        self.kernel.set_checkpoint_metadata(metadata)

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

    # -- calibration (the cMD pre-run + parameter selection) ----------------

    def _calibrate(self) -> None:
        """Zero-strength MD in ``calibration_interval`` chunks, sampling
        each channel's target energy from ``boost_potentials()`` — the
        integrator's own P globals, so calibration samples the exact
        quantity production boosts (method-side pure logic, not a kernel
        seam; ADR-0005 rejected integrator-side Welford windows)."""
        stage = GamdCalibration(self)
        while not stage.done:
            stage.advance(self.calibration_interval)

    def complete_calibration(self, samples, *, moments=None, persist=True):
        """Select and persist calibrated parameters from completed cMD samples."""
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
        self.parameter_history.append({"step": int(self.kernel.current_step), "channels": calibration})
        if persist:
            self._persist_calibration()
        self._push_calibration()

    def _persist_calibration(self):
        if self.sink is not None and self.calibration:
            self.sink.write_bytes(CALIBRATION_FILENAME, json.dumps({
                "sigma0": self.sigma0, "energy_unit": "kJ/mol", "channels": self.calibration,
                "parameter_history": self.parameter_history,
            }, indent=2, sort_keys=True).encode("utf-8"))

    def _load_calibration(self) -> None:
        """Resume path: load parameters from gamd_calibration.json — the
        ONE parameter source for fresh and resumed runs alike — instead
        of re-calibrating."""
        if self.sink is None:
            raise ValueError(
                f"continue_md gamd needs a sink to load "
                f"{CALIBRATION_FILENAME} from")
        data = json.loads(
            self.sink.read_bytes(CALIBRATION_FILENAME).decode("utf-8"))
        history = data.get("parameter_history")
        if history is None:
            # Older calibration artifacts did not record a revision step.
            # Their preparation sidecar is the authoritative completion step.
            if self.sink.exists("gamd_preparation.json"):
                completion = json.loads(self.sink.read_bytes("gamd_preparation.json"))["calibration_end"]
            else:
                completion = max(1, self.calibration_steps // self.calibration_interval) * self.calibration_interval
            history = [{"step": completion, "channels": data["channels"]}]
        self.parameter_history = [revision for revision in history
                                  if revision["step"] <= self.kernel.current_step]
        if not self.parameter_history:
            # Sidecars may be ahead of output.ckpt. Resume conventional MD
            # from its trimmed preparation samples, never future parameters.
            self.calibration = {}
            raise FileNotFoundError("no completed GaMD calibration at the restored checkpoint")
        channels = self.parameter_history[-1]["channels"]
        missing = set(self.channels) - set(channels)
        if missing:
            raise ValueError(
                f"saved gamd calibration lacks channel(s) {sorted(missing)} "
                f"(found {sorted(channels)}): the gamd channel layout "
                f"changed between runs")
        self.calibration = {label: channels[label] for label in self.channels}

    def _push_calibration(self) -> None:
        """Push (threshold, k) per channel via ``set_boost_param``; pushing
        over a restored checkpoint is idempotent (resume re-pushes, it
        does not re-calibrate)."""
        for label, params in self.calibration.items():
            self.kernel.set_boost_param(label, "threshold", params["threshold"])
            self.kernel.set_boost_param(label, "k", params["k"])


class GamdStages:
    """Absolute-step preparation with a replayable statistics/parameter journal.

    The cMD and adaptive windows use population variance. Adaptive statistics
    start a new cumulative window after boost preparation; production never
    updates parameters. Every transition completes before a caller checkpoints.
    """

    filename = "gamd_stages.json"

    def __init__(self, run, *, resumed=False):
        self.run = run
        settings = run.settings
        self.lengths = [int(settings["preparation_steps"]), run.calibration_steps,
                        int(settings["boost_preparation_steps"]),
                        int(settings["equilibration_steps"])]
        if any(n < 0 for n in self.lengths):
            raise ValueError("GaMD stage lengths must be nonnegative")
        self.interval = int(settings["adaptation_interval"] or run.calibration_interval)
        if self.interval < 1:
            raise ValueError("gamd.adaptation_interval must be positive")
        self.fingerprint = hashlib.sha256(json.dumps(
            {"settings": settings, "channels": list(run.channels.items())},
            sort_keys=True, default=list).encode()).hexdigest()
        self.start = int(run.kernel.current_step)
        run.control_history = [{"step": self.start, "sigma0": run.sigma0,
                                "revision": run.control_revision}]
        self.records = []
        if resumed:
            if run.sink is None or not run.sink.exists(self.filename):
                raise FileNotFoundError("staged GaMD resume requires gamd_stages.json")
            saved = json.loads(run.sink.read_bytes(self.filename))
            if saved["fingerprint"] != self.fingerprint:
                raise ValueError("GaMD ordered configuration changed since checkpoint")
            self.start = saved["start_step"]
            run.control_history = [r for r in saved["control_history"]
                                   if r["step"] <= run.kernel.current_step
                                   and r["revision"] <= run.control_revision]
            if not run.control_history:
                raise ValueError("GaMD checkpoint has no matching calibration control state")
            control = run.control_history[-1]
            if control["revision"] != run.control_revision or control["sigma0"] != run.sigma0:
                raise ValueError("GaMD control history disagrees with checkpoint metadata")
            for entry in run.settings["boosts"] or ():
                if "sigma0" not in entry:
                    run.channel_settings[entry["label"]]["sigma0"] = run.sigma0
            self.records = [r for r in saved["samples"] if r["step"] <= run.kernel.current_step]
            run.parameter_history = [r for r in saved["parameter_history"]
                                     if r["step"] <= run.kernel.current_step]
            if run.parameter_history:
                run.calibration = run.parameter_history[-1]["channels"]
        self.moments = {phase: {} for phase in ("statistics", "equilibration")}
        for record in self.records:
            self._accumulate(record)
        self.ends = np.cumsum([self.start] + self.lengths).tolist()[1:]
        if self.start + sum(self.lengths) > int(run.plan.steps):
            raise ValueError("gamd stage lengths exceed plan.steps (absolute final step)")
        if resumed and run.kernel.current_step >= self.ends[1] and not run.calibration:
            raise ValueError("GaMD checkpoint has no matching completed calibration state")
        self.persist()

    @property
    def phase(self):
        now = self.run.kernel.current_step
        for name, end in zip(("preparation", "statistics", "boost_preparation", "equilibration"), self.ends):
            if now < end:
                return name
        return "production"

    @property
    def done(self):
        return self.phase == "production"

    def _accumulate(self, record):
        moments = self.moments[record["phase"]]
        for label, value in record["energies"].items():
            m = moments.setdefault(label, {"count": 0, "mean": 0., "m2": 0., "min": value, "max": value})
            m["count"] += 1
            delta = value - m["mean"]
            m["mean"] += delta / m["count"]
            m["m2"] += delta * (value - m["mean"])
            m["min"], m["max"] = min(m["min"], value), max(m["max"], value)

    def persist(self):
        if self.run.sink is not None:
            self.run._persist_calibration()
            self.run.sink.write_bytes(self.filename, json.dumps({
                "schema_version": 1, "fingerprint": self.fingerprint,
                "start_step": self.start, "stage_ends": self.ends,
                "phase": self.phase, "step": int(self.run.kernel.current_step),
                "samples": self.records,
                "sample_counts": {phase: sum(r["phase"] == phase for r in self.records)
                                  for phase in ("statistics", "equilibration")},
                "parameter_history": self.run.parameter_history,
                "current_parameters": self.run.calibration,
                "runtime_sigma0": self.run.sigma0,
                "control_history": self.run.control_history,
            }, sort_keys=True).encode())

    def advance(self, max_steps):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        kernel = self.run.kernel
        cap = min(self.ends[-1], kernel.current_step + max_steps)
        while kernel.current_step < cap:
            phase = self.phase
            idx = ("preparation", "statistics", "boost_preparation", "equilibration").index(phase)
            origin = self.start if idx == 0 else self.ends[idx - 1]
            end = self.ends[idx]
            interval = self.interval if phase == "equilibration" else self.run.calibration_interval
            next_sample = origin + ((kernel.current_step - origin) // interval + 1) * interval
            target = self.run.next_exchange_boundary(min(cap, end, next_sample))
            kernel.step(target - kernel.current_step)
            self.run.exchange_at_boundary()
            sample = phase in ("statistics", "equilibration") and (target == end or target == next_sample)
            if sample:
                readings = kernel.boost_potentials()
                values = {label: float(readings[label].energy) for label in self.run.channels}
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError("nonfinite GaMD calibration energy")
                record = {"step": target, "phase": phase, "energies": values}
                self.records.append(record)
                self._accumulate(record)
                if phase == "equilibration" or target == end:
                    moments = self.moments[phase]
                    # A single adaptive sample has no usable variance. Retain
                    # cMD parameters until the window has at least two samples.
                    if phase == "statistics" or next(iter(moments.values()))["count"] >= 2:
                        self.run.complete_calibration({label: () for label in moments},
                                                      moments=moments, persist=False)
        # Flush once per bounded driver turn, and on explicit checkpoint(),
        # rather than rewriting the cumulative journal after every sample.
        self.persist()
        return kernel.current_step


class GamdCalibration:
    """Bounded cMD preparation preserving the original sample boundaries."""

    def __init__(self, run):
        self.run = run
        self.interval = run.calibration_interval
        self.start_step = run.kernel.current_step
        self.total = self.start_step + max(1, run.calibration_steps // self.interval) * self.interval
        self.samples = {label: [] for label in run.channels}
        self.sample_steps = []
        if getattr(run.plan, "continue_md", False) and run.sink is not None:
            data = json.loads(run.sink.read_bytes("gamd_preparation.json"))
            self.start_step, self.total = data["start_step"], data["calibration_end"]
            keep = [i for i, step in enumerate(data["sample_steps"])
                    if step <= run.kernel.current_step]
            self.sample_steps = [data["sample_steps"][i] for i in keep]
            self.samples = {label: [data["samples"][label][i] for i in keep]
                            for label in run.channels}
        if run.sink is not None:
            run.sink.write_bytes("gamd_preparation.json", json.dumps({
                "sample_steps": self.sample_steps, "samples": self.samples,
                "start_step": self.start_step, "calibration_end": self.total,
            }, sort_keys=True).encode())

    @property
    def done(self):
        return bool(self.run.calibration)

    def advance(self, max_steps):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        cap = min(self.total, self.run.kernel.current_step + max_steps)
        while self.run.kernel.current_step < cap:
            now = self.run.kernel.current_step
            target = self.run.next_exchange_boundary(min(
                cap, self.start_step + ((now - self.start_step) // self.interval + 1) * self.interval))
            self.run.kernel.step(target - now)
            self.run.exchange_at_boundary()
            if (target - self.start_step) % self.interval == 0:
                readings = self.run.kernel.boost_potentials()
                self.sample_steps.append(target)
                for label in self.samples:
                    self.samples[label].append(float(readings[label].energy))
        if self.run.sink is not None:
            self.run.sink.write_bytes("gamd_preparation.json", json.dumps({
                "sample_steps": self.sample_steps, "samples": self.samples,
                "start_step": self.start_step, "calibration_end": self.total,
            }, sort_keys=True).encode())
        if self.run.kernel.current_step >= self.total:
            self.run.complete_calibration(self.samples)
        return self.run.kernel.current_step


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
    ``boost_kjmol`` is the ``<label>__boost`` column of ``gamd.tsv``;
    unbiased expectations follow the weight w = exp(+β·ΔV).
    """
    from neomd.analysis.reweight import reweight_expectation

    return reweight_expectation(values, boost_kjmol, temperature)


# ---------------------------------------------------------------------------
# schema + registration (the knowledge triple)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": {
        "gamd": ("mapping: mode (total | dual), sigma0 (kJ/mol, default "
                 "6.0), calibration_steps, calibration_interval, frequency "
                 "(gamd.tsv cadence), channels (optional explicit "
                 "[{label, groups}] — LiGaMD group definitions)"),
        "steps": ("int, the FINAL step of the whole run — the calibration "
                  "pre-run advances the same counter (plan-level key)"),
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "gamd.boosts": ("ordered [{label, selectors:[{atom_indices,force_types}], sigma0?, "
                        "threshold_policy?, energy_unit?}] with optional final {label,remaining:true}; "
                        "zero-based any-atom match, first-defined owns each term; restraints excluded"),
        "gamd.threshold_policy": "upper (default, lower fallback) or lower, independently overridable per boost",
        "gamd.energy_unit": "kJ/mol (default) or kcal/mol; kcal/mol is explicitly multiplied by 4.184",
        "gamd.preparation_steps": "nonnegative conventional-MD preparation steps (default 0)",
        "gamd.boost_preparation_steps": "nonnegative fixed initial-boost preparation steps (default 0)",
        "gamd.equilibration_steps": "nonnegative adaptive boosted equilibration steps (default 0); production freezes parameters",
        "gamd.adaptation_interval": "positive adaptation sampling interval; defaults to calibration_interval",
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
        "continue_md": ("bool; restore output.ckpt, trim gamd.tsv and push "
                        "the saved gamd_calibration.json parameters (the "
                        "calibration pre-run never re-runs)"),
        "output.report_gamd": ("bool, default true — the gamd.tsv boost "
                               "trace switch (driver._TAPE_SWITCHES)"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'gamd'."""
    return GamdRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", LABEL, Method(schema=SCHEMA, prepare=_prepare))
