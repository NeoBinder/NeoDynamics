"""Gaussian-accelerated MD (GaMD) — a method knowledge triple (issue #10, W2-b).

Physics (Miao/Feher/McCammon, JCTC 2015; Miao 2016 dual boost; Miao/
Bhattarai/Wang JCTC 2020 LiGaMD; Copeland/Miao et al. JPCB 2022
gamd-openmm), ported onto the v2 seams per ADR-0005:

* the boost potential ΔV(P) = ½·k·(E−P)² (applied while P < E) depends on
  the boosted region's OWN potential energy, so the biased force is a
  SCALED system force — installed through the kernel's ``BoostOps``
  capability (``install_boost`` / ``set_boost_param`` / ``boost_potentials``),
  not an additive ``BiasIR`` (mathematically impossible; ADR-0005);
* channels are installed at ZERO strength in ``prepare()`` — the same
  pre-Context discipline as ``install_bias`` — and the calibrated
  (threshold E, harmonic k) values are pushed live afterwards, the
  ``BiasParamOps`` pattern applied to channels.  ``mode: total`` installs
  one ``total`` channel (``groups == ()``, the system energy); ``mode:
  dual`` adds a ``dihedral`` channel over the system's torsion force
  groups (discovered through the kernel's duck-typed
  ``torsion_force_groups()``; the openmm adapter isolates torsion forces
  into a fresh group pre-Context via ``pick_free_force_group``, the fake
  reports installed torsion biases — group ids stay opaque ints);
  ``channels: [{label, groups}]`` defines explicit channels (LiGaMD: point
  them at the ligand dihedral / ligand-nonbonded groups of a system whose
  XML already separates those interactions into their own force groups);
* the calibration pre-run is METHOD-side pure logic, not a kernel seam
  (ADR-0005 rejected integrator-side Welford windows): zero-strength MD in
  ``calibration_interval`` chunks, each chunk's per-channel target energy
  read from ``boost_potentials()`` — the integrator's own P globals, so
  calibration samples the exact quantity production boosts — then the
  literature (E, k) selection:

  - lower bound:  E = Vmax, k0 = min(1, (σ0/σV)·(Vmax−Vmin)/(Vmax−Vavg))
  - upper bound:  k0 = (1−σ0/σV)·(Vmax−Vmin)/(Vavg−Vmin), E = Vmin +
    (Vmax−Vmin)/k0  (used when its k0 lands in (0, 1], else the lower
    bound; a degenerate sample range — σV = 0 — keeps the channel at zero
    strength, loudly logged);

  with the effective harmonic constant k = k0/(Vmax−Vmin) ∈ (0, 1/(Vmax−Vmin)],
  which bounds the force scaling s = 1 − k(E−P) to [0, 1] (forces never
  flip).  Parameters land in ``gamd_calibration.json`` — the ONE parameter
  source for fresh and resumed runs alike (resume pushes from it; pushing
  over a checkpoint is idempotent);
* ``gamd.tsv`` — the boost trace: per channel ΔV (the reweighting
  observable), the target energy P and the force scale, read through
  ``boost_potentials()`` by :class:`neomd.probes.GamdProbe` (switch-gated
  by ``output.report_gamd``, default on, trimmed on resume like every
  other tape);
* reweighting: unbiased expectations follow the Tiwary–Parrinello weight
  w = exp(+β·ΔV) — exactly :func:`neomd.analysis.reweight.tp_weights` /
  :func:`neomd.analysis.reweight.reweight_expectation` fed the
  ``<label>__boost`` column (helpers below parse ``gamd.tsv`` so the
  analysis subpackage stays the single reweighting definition point).

Step accounting (documented): the calibration pre-run advances the kernel's
step counter — ``plan.steps`` is the FINAL step of the whole run
(calibration + boosted production), the same absolute-step convention
``resume`` uses.

This module never imports openmm (methods stay kernel-agnostic; units are
the port's kJ/mol conventions throughout).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd.kernel.port import (
    BoostChannelIR,
    BoostOps,
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


def _channel_params(samples: np.ndarray, sigma0: float) -> dict:
    """Literature (E, k) selection from calibration samples (kJ/mol).

    Returns ``{"threshold": E, "k": k, "vmax": ..., "vmin": ...,
    "vavg": ..., "sigma": ...}``; a degenerate sample range (σV = 0, the
    system's energy does not vary — e.g. a frozen fixture) keeps the
    channel at ZERO strength rather than dividing by zero.
    """
    vmax = float(np.max(samples))
    vmin = float(np.min(samples))
    vavg = float(np.mean(samples))
    sigma = float(np.std(samples))
    result = {"threshold": 1e99, "k": 0.0, "vmax": vmax, "vmin": vmin,
              "vavg": vavg, "sigma": sigma,
              "bound": None if sigma == 0.0 else "lower"}
    if sigma == 0.0 or vmax <= vmin or not (vmax > vavg > vmin):
        return result  # no usable spread: stay at zero strength
    k0_lower = min(1.0, (sigma0 / sigma) * (vmax - vmin) / (vmax - vavg))
    k0_upper = (1.0 - sigma0 / sigma) * (vmax - vmin) / (vavg - vmin)
    if 0.0 < k0_upper <= 1.0:
        # upper bound (threshold above the sample mean): E from k0
        threshold = vmin + (vmax - vmin) / k0_upper
        k0 = k0_upper
        result["bound"] = "upper"
    else:
        # lower bound (threshold at Vmax): gamd-openmm's E = Vmax form
        threshold = vmax
        k0 = k0_lower
        result["bound"] = "lower"
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
        if settings["mode"] not in _VALID_MODES and not settings["channels"]:
            raise ValueError(
                f"gamd.mode must be one of {_VALID_MODES} (or define "
                f"gamd.channels explicitly), got {settings['mode']!r}")
        sigma0 = float(settings["sigma0"])
        if sigma0 <= 0.0:
            raise ValueError(f"gamd.sigma0 must be > 0, got {sigma0}")
        self.settings = settings
        self.sigma0 = sigma0
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
        if settings["channels"]:
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
        if resumed:
            self._load_calibration()
        if self.calibration:
            # fresh-with-saved-calibration or resume: push, never re-run
            # (pushing over a checkpoint is idempotent — ADR-0005)
            self._push_calibration()
        else:
            if resumed:
                raise FileNotFoundError(
                    f"cannot continue gamd: {CALIBRATION_FILENAME} not "
                    f"found in the sink (a resumed GaMD run needs the "
                    f"parameters its calibration pre-run produced)")
            self._calibrate()

        tapes: dict = {}
        if self.sink is not None:
            tapes[TAPE_FILENAME] = GamdProbe(
                self.sink,
                interval=self.frequency,
                labels=list(self.channels),
                append=resumed and TAPE_FILENAME in resume_plan.trims,
            )
        return PreparedMethod(
            on_step=None,
            fgroups={},  # GaMD installs no bias force
            resume_plan=resume_plan,
            tapes=tapes,
            finish=self._finish,
        )

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
        """Zero-strength MD in chunks, sampling each channel's P."""
        samples: dict[str, list[float]] = {label: []
                                           for label in self.channels}
        n_chunks = max(1, self.calibration_steps // self.calibration_interval)
        for chunk in range(n_chunks):
            self.kernel.step(self.calibration_interval)
            readings = self.kernel.boost_potentials()
            for label in self.channels:
                samples[label].append(float(readings[label].energy))
        calibration = {}
        for label, values in samples.items():
            params = _channel_params(np.asarray(values, dtype=np.float64),
                                     self.sigma0)
            calibration[label] = params
            if params["k"] == 0.0:
                self.log.warning(
                    "gamd channel %r kept at zero strength (sigmaV=%.6g "
                    "kJ/mol — the energy did not vary enough to calibrate)",
                    label, params["sigma"])
            else:
                self.log.info(
                    "gamd channel %r calibrated (%s bound): E=%.6g kJ/mol, "
                    "k0*k=%.6g 1/(kJ/mol) from %d samples "
                    "(Vmax=%.6g Vmin=%.6g sigmaV=%.6g)",
                    label, params["bound"], params["threshold"],
                    params["k"], len(values), params["vmax"],
                    params["vmin"], params["sigma"])
        self.calibration = calibration
        if self.sink is not None:
            self.sink.write_bytes(
                CALIBRATION_FILENAME,
                json.dumps({"sigma0": self.sigma0, "channels": calibration},
                           indent=2, sort_keys=True).encode("utf-8"))
        self._push_calibration()

    def _load_calibration(self) -> None:
        if self.sink is None:
            raise ValueError(
                f"continue_md gamd needs a sink to load "
                f"{CALIBRATION_FILENAME} from")
        data = json.loads(
            self.sink.read_bytes(CALIBRATION_FILENAME).decode("utf-8"))
        channels = data["channels"]
        missing = set(self.channels) - set(channels)
        if missing:
            raise ValueError(
                f"saved gamd calibration lacks channel(s) {sorted(missing)} "
                f"(found {sorted(channels)}): the gamd channel layout "
                f"changed between runs")
        self.calibration = {label: channels[label] for label in self.channels}

    def _push_calibration(self) -> None:
        for label, params in self.calibration.items():
            self.kernel.set_boost_param(label, "threshold", params["threshold"])
            self.kernel.set_boost_param(label, "k", params["k"])


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
    — the reweighting definition stays in the analysis subpackage (W1-a).
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
        "gamd.channels": ("explicit channel definitions [{label, groups}] "
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
