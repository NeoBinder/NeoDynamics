"""OPES — on-the-fly probability enhanced sampling, a method knowledge triple.

Structural mirror of :mod:`neomd.methods.metadynamics`: the bias is a
``BiasIR(kind="CustomCVTableForce")`` installed through
``kernel.install_bias``, the update cycle rides the driver's ``on_step``
hook (``on_step_interval = opes_set.pace`` — OPES updates its averaged
quantities on a cadence of PACE steps, far slower than the MD step), and
live bias access goes through ``kernel.bias_ops()``
(cv_values / bias_energy / update_table).  The fake kernel runs the whole
OPES math deterministically (no RNG anywhere — the compression below is
nearest-kernel, not random).

Physics — Invernizzi & Parrinello, JCTC 16, 7113 (2020) and Invernizzi,
Piaggi & Parrinello, JCTC 18, 3988 (2022), cross-checked against the PLUMED
reference implementation (``src/opes/OPESmetad.cpp``, LGPL, consulted as
documentation only — no code copied):

* weighted KDE of the unbiased marginal
  ``P_n(s) = Σ_k w_k G(s, s_k) / Σ_k w_k``, ``w_k = exp(β V_{k-1}(s_k))``;
  OPES-explore instead estimates the SAMPLED (biased) distribution with
  ``w_k = 1`` (plain KDE).
* ``V_n(s) = (1 - 1/γ) (1/β) log( P_n(s)/Z_n + ε )`` (standard) and
  ``V_n(s) = (γ - 1) (1/β) log( p^WT_n(s)/Z_n + ε )`` (explore).
* ``Z_n = (1/|Ω_n|) ∫_{Ω_n} P_n ds`` — the KDE averaged over the EXPLORED
  CV region (the exit-time fix).
* ``ε = exp( -ΔE / (k_B T (1 - 1/γ)) )``, ``γ = ΔE / (k_B T)`` — BARRIER
  derives both; no biasFactor/height keys exist in the schema.
* adaptive bandwidth
  ``σ_i(n) = σ_i(0) [N_eff (d+2)/4]^{-1/(d+4)}``,
  ``N_eff = (Σ w_k)^2 / Σ w_k^2`` (Silverman shrinking; default ON).
* kernel compression: append a kernel only when a NEW CV region is
  sampled, otherwise merge it into the existing set.
* input parameters: initial kernel width, PACE, BARRIER — here the width
  is each colvar's ``biasWidth`` (the same key metadynamics uses) and
  PACE/BARRIER live in ``opes_set``.
* explore-mode FES estimator ``F_n(s) = -γ (1/β) log p^WT_n(s)``; the
  standard-mode estimator is ``F_n(s) = -(1/β) log P_n(s)`` (the two are
  equivalent in standard mode, bias-converted vs KDE).

Implementation details from the papers / PLUMED:

* compression is NEAREST-KERNEL within a Mahalanobis threshold
  ``COMPRESSION_THRESHOLD`` (default 1) in units of the EXISTING kernel's
  sigma: merge if ``Σ_i (Δs_i / σ_i)^2 < threshold^2`` (σ of the old
  kernel), else append; merging sums heights, takes the weighted-average
  center and the weighted second moment of sigma, and then RETRIES
  recursively (the merged kernel may now be mergeable with another).
* kernels are truncated at ``KERNEL_CUTOFF`` sigmas:
  ``G → (exp(-||Δ||²/2) - exp(-cutoff²/2))`` beyond which they are 0 —
  the sum stays finite and compactly supported.
* ``Z_n`` is evaluated by substituting the integral with a sum over the
  COMPRESSED kernel positions (paper §"Making the Bias"):
  ``Z = (1/N_ker) Σ_k [ Σ_j G_j(s_k) ] / KDEnorm``.
* γ = ΔE/(k_BT) is derived from BARRIER (PLUMED's default), ``Σw`` is
  seeded with ``w_0 = ε^{prefactor}`` at ``counter = 1`` (a virtual
  unbiased kernel that keeps N_eff finite at the start), and each kernel's
  height is multiplied by ``∏_i σ_i(0)/σ_i(n)`` — the ``1/(√2π σ)``
  Gaussian normalization up to a constant that Z absorbs.
* deposit cadence PACE fires ONE kernel deposit + Z refresh + table push
  per event; nothing is updated per-MD-step.

Deliberate deviations (none contradicts the physics):

* the ledger: ``kernels.npz`` carries every PRE-compression deposit
  ``{steps, positions (n,ncv), sigmas (n,ncv), heights (n,),
  logweights (n,)}`` in kernel CV units (nm / radian — the same space
  metadynamics' hills live in).  Like ``hills.npz`` it is method STATE, not
  reporting: it is written by the deposit hook itself (a probe fires
  BEFORE ``on_step`` at a shared boundary, so a probe-written ledger would
  lag one deposit behind and break bit-exact resume), which is also why it
  is NOT switch-gated through ``driver._TAPE_SWITCHES`` — the resume ledger
  is physics, exactly like ``hills.npz``.
* the deposit weight reads the bias through the TABLE the simulated system
  feels (``bias_ops().bias_energy``), not a parallel continuous evaluation —
  one definition of the bias on this seam.
* PLUMED widens a user-provided SIGMA by √γ in explore mode; we keep the
  declared ``biasWidth`` verbatim as σ(0) in both modes — it is the initial
  kernel width.
* resume replays the (trimmed) ledger through the SAME deposit math and
  recomputes Z ONCE from the final compressed set (PLUMED's restart rule) —
  deterministic, so a resumed run's kernels and table are bit-identical to
  an uninterrupted run's.
"""

from __future__ import annotations

import datetime
import io
import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from neomd.kernel.port import (
    CVIR,
    BiasIR,
    GridSpec,
    TableSpec,
    cv_is_angular,
    to_canonical,
)
from neomd.methods.metadynamics import MOLAR_GAS_CONSTANT_R_KJ
from neomd.registry import register

__all__ = [
    "Method",
    "MethodResult",
    "OPESRun",
    "LABEL",
    "KERNELS_FILENAME",
    "FES_FILENAME",
    "MODES",
]

LABEL = "opes"
KERNELS_FILENAME = "kernels.npz"
FES_FILENAME = "fes.tsv"

#: the two OPES variants (well-tempered target vs uniform-exploration target)
MODES = ("standard", "explore")

#: element budget of one kernel-evaluation broadcast block (kernels x points)
_KERNEL_BUDGET = 2_000_000

LOG = logging.getLogger("neomd.methods.opes")


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method").

    ``prepare`` has the drive() dispatch signature
    ``prepare(kernel=..., plan=..., sink=..., logger=...) ->
    neomd.driver.PreparedMethod`` — it installs the method's biases, plans
    its resume, and builds its tape probes; the DRIVER runs the loop and
    owns reporting (driver.run_prepared_method).
    """

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one OPES run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count
    fgroup: int  # force-group id of the installed table bias
    n_deposits: int  # kernel deposits (incl. replayed ones after a resume)
    n_kernels: int  # compressed kernel-set size (<= n_deposits)
    zed: float  # the final Z_n normalization (1.0 with no_zed)
    rct: float  # kT*log(Σw/n) — the c(t)-analogue reweighting constant
    fes_sum: float  # sum of the free-energy grid (kJ/mol) — a drift sentinel
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# grid standardization — shared with metadynamics (degrees -> radians for
# angular CVs through the shared port table)
# ---------------------------------------------------------------------------


def _is_angular(cv: CVIR) -> bool:
    return cv_is_angular(cv)


def _standardize(value: float, cv: CVIR) -> float:
    if _is_angular(cv):
        return to_canonical(value, "deg")
    return float(value)


def _grid_unit(cv: CVIR) -> str:
    return "rad" if _is_angular(cv) else "nm"


def _make_evaluator(entry, cv: CVIR) -> Callable:
    """(positions, masses) -> float closure around one colvars entry."""

    def evaluate(positions, masses):
        return entry.evaluate(positions, masses, cv)

    return evaluate


# ---------------------------------------------------------------------------
# the compressed kernel set
# ---------------------------------------------------------------------------


class _Kernel:
    """One compressed KDE kernel: weight (height), center, per-axis sigma."""

    __slots__ = ("height", "center", "sigma")

    def __init__(self, height: float, center: np.ndarray, sigma: np.ndarray):
        self.height = float(height)
        self.center = np.array(center, dtype=np.float64)
        self.sigma = np.array(sigma, dtype=np.float64)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class OPESRun:
    """One OPES execution over a kernel (standard or explore mode).

    Construct directly for artifact access (``get_free_energy`` /
    ``write_fes`` survive the run).  ``prepare()`` is the registry entry
    drive() dispatches; ``run()`` is the direct-construction convenience
    (prepare + the driver's method loop) returning the :class:`MethodResult`.
    """

    def __init__(self, kernel, plan, sink=None, logger=None):
        self.kernel = kernel
        self.plan = plan
        self.sink = sink
        self.log = LOG if logger is None else logger

        # -- opes_set (pace + barrier + mode + tuning knobs) -----------------
        opes = dict(getattr(plan, "opes_set", None) or {})
        missing = [key for key in ("pace", "barrier") if key not in opes]
        if missing:
            raise ValueError(
                f"opes_set requires {', '.join(missing)} (got keys {sorted(opes)})")
        self.mode = str(opes.get("mode", "standard"))
        if self.mode not in MODES:
            raise ValueError(
                f"opes_set.mode must be one of {list(MODES)}, got {self.mode!r}")
        self.explore = self.mode == "explore"
        self.pace = int(opes["pace"])
        if self.pace < 1:
            raise ValueError(f"opes_set.pace must be >= 1, got {self.pace}")
        self.barrier = float(opes["barrier"])
        if self.barrier <= 0.0:
            raise ValueError(
                f"opes_set.barrier must be > 0 (kJ/mol), got {self.barrier}")
        self.compression_threshold = float(opes.get("compression_threshold", 1.0))
        if self.compression_threshold < 0.0:
            raise ValueError(
                "opes_set.compression_threshold must be >= 0 "
                f"(0 disables merging), got {self.compression_threshold}")
        self.fixed_sigma = bool(opes.get("fixed_sigma", False))
        self.no_zed = bool(opes.get("no_zed", False))

        # -- BARRIER-derived constants (PLUMED default mapping) -------------
        self.temperature = float(getattr(plan, "temperature", 298.0))
        self.kt = MOLAR_GAS_CONSTANT_R_KJ * self.temperature  # kJ/mol
        self.bias_factor = self.barrier / self.kt  # gamma = ΔE / (k_B T)
        if self.bias_factor <= 1.0:
            raise ValueError(
                f"barrier {self.barrier} kJ/mol is below kT at "
                f"{self.temperature} K (gamma = {self.bias_factor:.3f} <= 1); "
                "raise opes_set.barrier")
        self.prefactor = (self.bias_factor - 1.0) if self.explore \
            else (1.0 - 1.0 / self.bias_factor)
        self.epsilon = math.exp(-self.barrier / (self.prefactor * self.kt))
        if "kernel_cutoff" in opes:
            self.cutoff = float(opes["kernel_cutoff"])
            if self.cutoff <= 0.0:
                raise ValueError(
                    f"opes_set.kernel_cutoff must be > 0, got {self.cutoff}")
        else:
            # PLUMED defaults: sqrt(2ΔE/(prefactor·kT)); explore needs the
            # wider sqrt(2ΔE/kT) "otherwise it is too small"
            self.cutoff = math.sqrt(
                2.0 * self.barrier / (self.kt if self.explore
                                      else self.prefactor * self.kt))
        self.cutoff2 = self.cutoff * self.cutoff
        self.val_at_cutoff = math.exp(-0.5 * self.cutoff2)
        self.threshold2 = self.compression_threshold ** 2
        if self.threshold2 != 0.0 and self.threshold2 >= self.cutoff2:
            raise ValueError(
                f"opes_set.compression_threshold ({self.compression_threshold}) "
                f"cannot be >= the kernel cutoff ({self.cutoff:.6f} sigma)")

        # -- collective variables through the cv registry -------------------
        colvar_cfg = dict(getattr(plan, "colvars", None) or {})
        if not colvar_cfg:
            raise ValueError("opes requires plan.colvars (1-3 entries)")

        import neomd.colvars  # noqa: F401  (import = cv registration)
        from neomd import registry

        self.cvs: list[tuple[str, CVIR, object]] = []  # (name, CVIR, entry)
        self.grids: list[GridSpec] = []  # kernel-standardized, config order
        for name, spec in colvar_cfg.items():
            spec = dict(spec)
            if "type" not in spec:
                raise ValueError(f"colvar {name!r} requires a 'type'")
            entry = registry.get("cv", spec["type"])  # KeyError w/ did-you-mean
            cv, grid = entry.make_cv(name, spec)
            gspec = GridSpec(
                minimum=_standardize(grid["min"], cv),
                maximum=_standardize(grid["max"], cv),
                width=_standardize(grid["width"], cv),
                bins=int(grid["bins"]),
                periodic=bool(grid["periodic"]),
            )
            if gspec.width <= 0.0:
                raise ValueError(
                    f"colvar {name!r}: biasWidth must be > 0 (it is the "
                    f"initial OPES kernel width)")
            self.cvs.append((name, cv, entry))
            self.grids.append(gspec)

        # σ(0): the declared biasWidths (the initial kernel width)
        self.sigma0 = np.array([g.width for g in self.grids], dtype=np.float64)
        self.ncv = len(self.grids)

        # -- KDE state (PLUMED init: counter 1 + a virtual unbiased kernel) -
        self.counter = 1
        self.sum_weights = 1.0 if self.no_zed else self.epsilon ** self.prefactor
        self.sum_weights2 = self.sum_weights * self.sum_weights
        self.kdenorm = self.sum_weights  # Σw (standard) / n (explore), per mode
        self.kernels: list[_Kernel] = []
        self.zed = 1.0

        # the deposit ledger (kernels.npz rows, pre-compression)
        self._steps: list[int] = []
        self._positions: list[list[float]] = []
        self._sigmas: list[list[float]] = []
        self._heights: list[float] = []
        self._logweights: list[float] = []
        self.fgroup: int | None = None

    # -- entry point --------------------------------------------------------

    def prepare(self):
        """Install the table bias, plan the resume, build the colvar tape.

        The driver runs the loop (driver.run_prepared_method) and owns
        reporting — the restraint tape is attached there, not here.
        """
        from neomd.driver import PreparedMethod

        var_names = ["cv%d" % i for i in range(len(self.grids))]
        table = TableSpec(
            cvs=[cv for _, cv, _ in self.cvs],
            grids=list(self.grids),
            initial=np.zeros(self._grid_shape).flatten(),
            label=LABEL,
        )
        bias = BiasIR(
            kind="CustomCVTableForce",
            energy="table(%s)" % ", ".join(var_names),
            table=table,
            label=LABEL,
        )
        self.fgroup = self.kernel.install_bias(bias)

        ops = self.kernel.bias_ops()
        needed = ("cv_values", "bias_energy", "update_table")
        if ops is None or any(not callable(getattr(ops, name, None))
                              for name in needed):
            raise NotImplementedError(
                f"kernel {self.kernel.name!r} does not provide bias_ops() "
                f"({', '.join(needed)}); opes cannot run on it")
        self._ops = ops

        # resume through the single owner (neomd.resume.plan_resume): the
        # kernel restore + tape trimming happen there, AFTER install_bias.
        from neomd.resume import plan_resume

        resume_plan = plan_resume(self.plan, self.kernel, self.sink)
        if resume_plan is not None:
            self._replay_ledger(resume_plan)

        tapes: dict = {}
        if self.sink is not None:
            from neomd.probes import ColvarProbe

            tapes["colvar.tsv"] = ColvarProbe(
                self.sink,
                interval=self.pace,
                cvs=[{"label": name, "evaluate": _make_evaluator(entry, cv)}
                     for name, cv, entry in self.cvs],
                masses=self.kernel.masses,
                append=resume_plan is not None
                and "colvar.tsv" in resume_plan.trims,
            )
        return PreparedMethod(
            on_step=self._deposit,
            on_step_interval=self.pace,
            fgroups={LABEL: [self.fgroup]},
            resume_plan=resume_plan,
            tapes=tapes,
            progress=lambda step: {KERNELS_FILENAME: int(step)},
            finish=self._finish,
        )

    def _finish(self, result) -> MethodResult:
        """End-of-run artifacts + the result drive() records."""
        from neomd.driver import CHECKPOINT_FILENAME

        self._save_kernels()
        if self.sink is not None:
            self.sink.write_bytes(CHECKPOINT_FILENAME, self.kernel.snapshot())
            try:
                self.write_fes(self.sink.path(FES_FILENAME))
            except NotImplementedError:
                pass  # filesystem-less sink (MemorySink): fes stays in memory
        return MethodResult(
            steps_done=result.steps_done,
            fgroup=self.fgroup,
            n_deposits=len(self._steps),
            n_kernels=len(self.kernels),
            zed=float(self.zed),
            rct=self.kt * math.log(self.sum_weights / self.counter),
            fes_sum=float(np.sum(self.get_free_energy())),
            positions_sha256=result.positions_sha256,
        )

    def run(self) -> MethodResult:
        """Direct-construction entry: prepare + the driver's method loop
        (drive() calls prepare() and runs the loop itself — both paths share
        the one definition, driver.run_prepared_method)."""
        from neomd.driver import run_prepared_method

        return run_prepared_method(self.kernel, self.plan, self.prepare(),
                                   sink=self.sink, logger=self.log)

    # -- resume ----------------------------------------------------------------

    def _replay_ledger(self, resume_plan) -> None:
        """Rebuild the kernel set + running averages from the ledger.

        Mirrors metadynamics' ``_replay_ledger``: the ledger (already
        trimmed to the checkpoint step by neomd.resume — the single owner)
        is replayed row by row through the SAME ``_add_kernel`` compression
        math, the weight sums are re-accumulated from the stored
        logweights, and Z is recomputed ONCE from the final compressed set
        (PLUMED's restart rule — Z is a function of the final state, so one
        recompute is exact).  The table is then pushed once, exactly where
        metadynamics pushes its replayed bias.
        """
        if self.sink is None:
            raise ValueError(
                f"continue_md needs a sink to load {KERNELS_FILENAME} from")
        try:
            data = self.sink.read_bytes(KERNELS_FILENAME)
        except (KeyError, FileNotFoundError):
            path = None
            try:
                path = self.sink.path(KERNELS_FILENAME)
            except NotImplementedError:
                pass
            raise FileNotFoundError(
                f"cannot continue opes: {KERNELS_FILENAME} not found"
                + (f" at {path}" if path else "")) from None
        with np.load(io.BytesIO(data)) as loaded:
            steps = np.asarray(loaded["steps"], dtype=np.int64)
            positions = np.asarray(loaded["positions"], dtype=np.float64)
            sigmas = np.asarray(loaded["sigmas"], dtype=np.float64)
            heights = np.asarray(loaded["heights"], dtype=np.float64)
            logweights = np.asarray(loaded["logweights"], dtype=np.float64)
        positions = positions.reshape(len(steps), self.ncv)
        sigmas = sigmas.reshape(len(steps), self.ncv)
        for step, position, sigma, height, logweight in zip(
                steps.tolist(), positions.tolist(), sigmas.tolist(),
                heights.tolist(), logweights.tolist()):
            self.counter += 1
            weight = math.exp(logweight)
            self.sum_weights += weight
            self.sum_weights2 += weight * weight
            self.kdenorm = float(self.counter) if self.explore \
                else self.sum_weights
            self._add_kernel(height, position, sigma)  # replay, no table push
            self._steps.append(step)
            self._positions.append(position)
            self._sigmas.append(sigma)
            self._heights.append(height)
            self._logweights.append(logweight)
        self._update_zed()
        self._ops.update_table(LABEL, self._bias_grid().flatten())
        self.log.info("Load bias FILE:%s (%d deposits replayed, %d kernels)",
                      KERNELS_FILENAME, len(steps), len(self.kernels))

    # -- the deposit cycle (one kernel + one Z refresh + one table push) -----

    def _deposit(self, step: int, view) -> None:
        """One OPES update, every ``pace`` steps (the PACE cadence)."""
        position = self._ops.cv_values(LABEL)
        energy = self._ops.bias_energy(LABEL)  # kJ/mol — the tabulated bias
        log_weight = energy / self.kt
        weight = math.exp(log_weight)
        self.log.info(
            "Starting OPES cycle %d at time %s (step %d, bias energy %.6f "
            "kJ/mol, %d kernels)",
            len(self._steps) + 1, datetime.datetime.now(), step, energy,
            len(self.kernels))

        # running averages (PLUMED update(): counter, Σw, Σw², N_eff)
        self.counter += 1
        self.sum_weights += weight
        self.sum_weights2 += weight * weight
        neff = (1.0 + self.sum_weights) ** 2 / (1.0 + self.sum_weights2)

        if self.explore:
            self.kdenorm = float(self.counter)
            base_height = 1.0  # plain KDE — the bias reweight never enters
            size = float(self.counter)  # N_eff is not relevant in explore
        else:
            self.kdenorm = self.sum_weights
            base_height = weight
            size = neff

        # σ(n) = σ(0) · [size·(d+2)/4]^{-1/(d+4)}  (paper Eq. 6)
        sigma = np.array(self.sigma0, dtype=np.float64)
        if not self.fixed_sigma:
            sigma = sigma * (size * (self.ncv + 2) / 4.0) ** (
                -1.0 / (4.0 + self.ncv))
        # the 1/(√2π σ) normalization up to the constant Z absorbs
        # (PLUMED comment: "we skip it altogether, but keep any other
        # sigma rescaling")
        height = base_height * float(np.prod(self.sigma0 / sigma))

        self._add_kernel(height, position, sigma.copy())
        self._update_zed()
        self._ops.update_table(LABEL, self._bias_grid().flatten())

        self._steps.append(int(step))
        self._positions.append([float(v) for v in position])
        self._sigmas.append([float(v) for v in sigma])
        self._heights.append(float(height))
        self._logweights.append(float(log_weight))
        self._save_kernels()

    # -- kernel-set arithmetic (PLUMED OPESmetad, transcribed) ---------------

    def _difference(self, i: int, a: float, b: float) -> float:
        """Signed a−b with the minimal-image convention on periodic axes."""
        grid = self.grids[i]
        if not grid.periodic:
            return a - b
        period = grid.maximum - grid.minimum
        delta = (a - b) % period
        if delta > period / 2.0:
            delta -= period
        return delta

    def _bring_in_domain(self, i: int, value: float) -> float:
        """Wrap a periodic-axis coordinate back into [min, max)."""
        grid = self.grids[i]
        if not grid.periodic:
            return value
        period = grid.maximum - grid.minimum
        return grid.minimum + (value - grid.minimum) % period

    def _mergeable(self, center, exclude: int) -> int | None:
        """Index of the NEAREST kernel within the compression threshold
        (Mahalanobis distance in units of the EXISTING kernel's sigma),
        excluding ``exclude`` (a kernel does not merge with itself);
        None when no kernel is close enough."""
        best: int | None = None
        best_norm2 = self.threshold2
        if self.threshold2 == 0.0:
            return None
        for index, kernel in enumerate(self.kernels):
            if index == exclude:
                continue
            norm2 = 0.0
            for i in range(self.ncv):
                dist_i = self._difference(i, center[i], kernel.center[i]) \
                    / kernel.sigma[i]
                norm2 += dist_i * dist_i
                if norm2 >= best_norm2:
                    break
            if norm2 < best_norm2:
                best_norm2 = norm2
                best = index
        return best

    def _merge_kernels(self, k1: _Kernel, k2: _Kernel) -> None:
        """Merge k2 into k1: heights add, the center is the height-weighted
        mean, sigma² is the matched second moment (PLUMED mergeKernels)."""
        height = k1.height + k2.height
        for i in range(self.ncv):
            c1, c2 = k1.center[i], k2.center[i]
            if self.grids[i].periodic:
                c1 = c2 + self._difference(i, k1.center[i], c2)  # fix PBC
            center_i = (k1.height * c1 + k2.height * c2) / height
            ss = (k1.height * (k1.sigma[i] ** 2 + c1 * c1)
                  + k2.height * (k2.sigma[i] ** 2 + c2 * c2)) / height \
                - center_i * center_i
            k1.center[i] = self._bring_in_domain(i, center_i)
            k1.sigma[i] = math.sqrt(ss)
        k1.height = height

    def _add_kernel(self, height: float, center, sigma) -> None:
        """Deposit one kernel: merge into the nearest close kernel (then
        recursively retry — the merged kernel may be mergeable with
        another), or append when a NEW CV region is sampled."""
        new = _Kernel(height, center, sigma)
        taker = self._mergeable(new.center, len(self.kernels))
        if taker is None:
            self.kernels.append(new)
            return
        self._merge_kernels(self.kernels[taker], new)
        giver = taker
        taker = self._mergeable(self.kernels[giver].center, giver)
        while taker is not None:
            if taker > giver:
                taker, giver = giver, taker
            self._merge_kernels(self.kernels[taker], self.kernels[giver])
            del self.kernels[giver]
            giver = taker
            taker = self._mergeable(self.kernels[giver].center, giver)

    # -- KDE / bias / Z evaluation ------------------------------------------

    @property
    def _grid_shape(self) -> tuple:
        """Table layout: the reversed-axis convention (last CV fastest)."""
        return tuple(g.bins for g in reversed(self.grids))

    def _grid_points(self) -> tuple:
        """(points (m, ncv) in config order, config-order shape)."""
        coords = [np.linspace(g.minimum, g.maximum, num=g.bins)
                  for g in self.grids]
        if self.ncv == 1:
            return coords[0].reshape(-1, 1), (self.grids[0].bins,)
        grids = np.meshgrid(*coords, indexing="ij")
        shape = tuple(g.bins for g in self.grids)
        return np.stack([g.ravel() for g in grids], axis=1), shape

    def _to_table_layout(self, values: np.ndarray) -> np.ndarray:
        """Config-order point values -> the reversed-axis table array, with
        the periodic seam (``dist[-1] = dist[0]`` in metadynamics'
        ``_add_gaussian``): a periodic axis's LAST grid point must carry the
        FIRST's value — the KDE is periodic by construction up to floating
        point, and openmm's periodic spline demands exact equality there."""
        _, shape = self._grid_points()
        result = np.asarray(values, dtype=np.float64).reshape(shape)
        for i, grid in enumerate(self.grids):
            if grid.periodic:
                first = [slice(None)] * self.ncv
                last = [slice(None)] * self.ncv
                first[i], last[i] = 0, grid.bins - 1
                result[tuple(last)] = result[tuple(first)]
        if self.ncv == 1:
            return result
        return np.transpose(result, axes=tuple(range(self.ncv - 1, -1, -1)))

    def _kde_at(self, points: np.ndarray) -> np.ndarray:
        """Σ_k h_k·G(s_k, points) — the raw (unnormalized) kernel sum with
        the cutoff truncation, evaluated at every point; (m,) kJ/mol-scale."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        prob = np.zeros(points.shape[0], dtype=np.float64)
        if not self.kernels:
            return prob
        periods = np.array([g.maximum - g.minimum for g in self.grids])
        periodic = np.array([g.periodic for g in self.grids])
        n_points = points.shape[0]
        chunk = max(1, _KERNEL_BUDGET // max(1, self.ncv * n_points)
                    if n_points else 1)
        for lo in range(0, len(self.kernels), chunk):
            block = self.kernels[lo:lo + chunk]
            centers = np.array([k.center for k in block])  # (c, ncv)
            sigmas = np.array([k.sigma for k in block])    # (c, ncv)
            heights = np.array([k.height for k in block])  # (c,)
            dist = points[None, :, :] - centers[:, None, :]  # (c, m, ncv)
            if periodic.any():
                dist = np.where(periodic[None, None, :],
                                (dist + periods[None, None, :] / 2.0)
                                % periods[None, None, :]
                                - periods[None, None, :] / 2.0, dist)
            norm2 = ((dist / sigmas[:, None, :]) ** 2).sum(axis=2)  # (c, m)
            within = norm2 < self.cutoff2
            values = heights[:, None] * (np.exp(-0.5 * norm2)
                                         - self.val_at_cutoff)
            prob += (values * within).sum(axis=0)
        return prob

    def _log_argument(self, points: np.ndarray) -> np.ndarray:
        """P̃(points)/Z + ε (standard) / p^WT(points)/Z + ε (explore)."""
        return self._kde_at(points) / self.kdenorm / self.zed + self.epsilon

    def _update_zed(self) -> None:
        """Z = (1/N_ker) Σ_k [Σ_j G_j(s_k)] / KDEnorm — the KDE averaged over
        the COMPRESSED kernel positions (the explored-region proxy)."""
        if self.no_zed:
            self.zed = 1.0
            return
        if not self.kernels:
            self.zed = 1.0
            return
        centers = np.array([k.center for k in self.kernels])
        self.zed = float(self._kde_at(centers).sum()
                         / self.kdenorm / len(self.kernels))

    def _bias_grid(self) -> np.ndarray:
        """V_n on the deposition grid, reversed-axis shape (table layout).

        Before the first deposit there is no bias (PLUMED applies none
        until its first update): the table is exactly zero, which also
        keeps a resume from a pre-first-deposit checkpoint (e.g.
        ``checkpoint_interval < pace``) bit-identical to a fresh run —
        the epsilon-only formula would instead push a constant -barrier
        table everywhere."""
        if not self.kernels:
            return np.zeros(self._grid_shape)
        points, _ = self._grid_points()
        bias = self.kt * self.prefactor * np.log(self._log_argument(points))
        return self._to_table_layout(bias)

    # -- artifacts -----------------------------------------------------------------

    def _save_kernels(self) -> None:
        """kernels.npz — the deposit ledger (pre-compression rows)."""
        if self.sink is None:
            return
        steps = np.asarray(self._steps, dtype=np.int64)
        positions = np.asarray(self._positions, dtype=np.float64)
        sigmas = np.asarray(self._sigmas, dtype=np.float64)
        heights = np.asarray(self._heights, dtype=np.float64)
        logweights = np.asarray(self._logweights, dtype=np.float64)
        if positions.size == 0:
            positions = positions.reshape(0, self.ncv)
            sigmas = sigmas.reshape(0, self.ncv)
        buffer = io.BytesIO()
        np.savez(buffer, steps=steps, positions=positions, sigmas=sigmas,
                 heights=heights, logweights=logweights)
        self.sink.write_bytes(KERNELS_FILENAME, buffer.getvalue())

    def get_bias(self) -> np.ndarray:
        """The current bias table V_n (kJ/mol), reversed-axis grid shape."""
        return self._bias_grid()

    def get_free_energy(self) -> np.ndarray:
        """The mode's FES estimator on the deposition grid (kJ/mol).

        standard (equivalent to the bias conversion):
        ``F = -(1/β) log(P̃/Z + ε)``; explore: ``F = -γ (1/β) log(p^WT/Z + ε)``
        — both finite everywhere through the ε regularization.  With no
        kernels deposited yet the grid is exactly zero (no data, no FES).
        """
        if not self.kernels:
            return np.zeros(self._grid_shape)
        points, _ = self._grid_points()
        argument = self._log_argument(points)
        fes = -(self.bias_factor * self.kt if self.explore else self.kt) \
            * np.log(argument)
        return self._to_table_layout(fes)

    def write_fes(self, path) -> None:
        """Write ``fes.tsv`` — metadynamics' layout (same writer contract)."""
        fes = self.get_free_energy()
        coords = [np.linspace(g.minimum, g.maximum, num=g.bins)
                  for g in self.grids]
        header = "# " + "\t".join(
            f"{name} [{_grid_unit(cv)}]"
            for name, cv, _ in self.cvs) + "\tfes [kJ/mol]\n"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header)
            ncv = len(self.grids)
            for index in np.ndindex(fes.shape):
                # bias axis k <-> configured cv (ncv-1-k); row in config order
                row = [coords[j][index[ncv - 1 - j]] for j in range(ncv)]
                handle.write("\t".join(str(v) for v in row)
                             + f"\t{fes[index]}\n")


# ---------------------------------------------------------------------------
# schema + registration (the knowledge triple)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": {
        "colvars": ("mapping name -> colvar spec; each needs 'type' plus the "
                    "cv registry's keys — the grid (min/max/bins) is the bias "
                    "TABLE domain and biasWidth is σ(0), the initial kernel "
                    "width; 1-3 CVs"),
        "opes_set": ("mapping with pace (steps between OPES updates) and "
                     "barrier (expected free-energy barrier, kJ/mol); gamma, "
                     "epsilon and the kernel cutoff are derived from barrier "
                     "— no biasFactor/height keys exist (the spec's 3-input "
                     "design)"),
        "steps": "int, total steps (plan-level key)",
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "opes_set.mode": ("'standard' (default; well-tempered target, "
                          "convergence-oriented) or 'explore' (uniform-"
                          "exploration target, KDE of the sampled "
                          "distribution)"),
        "opes_set.compression_threshold": (
            "merge a new kernel into the nearest existing one when their "
            "distance is below this many sigmas (default 1; 0 disables "
            "merging)"),
        "opes_set.kernel_cutoff": (
            "truncate KDE kernels at this many sigmas (default derived from "
            "barrier: sqrt(2*barrier/(prefactor*kT)), explore uses "
            "sqrt(2*barrier/kT))"),
        "opes_set.fixed_sigma": (
            "bool, default false; true disables the N_eff bandwidth "
            "shrinking (sigma stays at biasWidth)"),
        "opes_set.no_zed": ("bool, default false; true sets Z_n = 1 (no "
                            "normalization over the explored region)"),
        "continue_md": ("bool; restore output.ckpt and replay kernels.npz "
                        "(trimmed to the checkpoint step) from the output "
                        "directory before running"),
        "output.*": ("output_dir + state/trajectory/checkpoint intervals "
                     "(plan-level; the colvar recorder always fires on "
                     "opes_set.pace)"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'opes'."""
    return OPESRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", "opes", Method(schema=SCHEMA, prepare=_prepare))
