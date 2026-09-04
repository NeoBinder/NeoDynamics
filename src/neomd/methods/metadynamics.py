"""Well-tempered metadynamics — a method knowledge triple (schema +
prepare + the deposition run).  See docs/methods/metadynamics.md.
Registers method: ``metadynamics``.  Never imports openmm (methods stay
kernel-agnostic; the bias rides ``kernel.install_bias`` / ``bias_ops()``).
"""

from __future__ import annotations

import datetime
import io
import logging
import math
from dataclasses import dataclass
from functools import reduce
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
from neomd.registry import register

__all__ = [
    "Method",
    "MethodResult",
    "MetadynamicsRun",
    "MOLAR_GAS_CONSTANT_R_KJ",
    "LABEL",
    "HILLS_FILENAME",
    "FES_FILENAME",
]

#: molar gas constant, kJ/(mol K) — bit-identical to openmm's
#: ``unit.MOLAR_GAS_CONSTANT_R.value_in_unit(kilojoule_per_mole/kelvin)``
#: (tests/v2/test_metadynamics.py asserts the equality; openmm itself is
#: never imported here — the methods layer is kernel-agnostic).
MOLAR_GAS_CONSTANT_R_KJ = 0.00831446261815324

#: molar gas constant, J/(mol K) — bit-identical to the raw ``_value`` of
#: openmm's ``unit.MOLAR_GAS_CONSTANT_R`` Quantity (verified empirically;
#: ``MOLAR_GAS_CONSTANT_R_KJ * 1000.0`` also lands on it exactly for this
#: mantissa, but the J-space raw value is what the Quantity-sequence port
#: below multiplies by, so it is spelled out).
_R_J_MOL_K = 8.31446261815324

LABEL = "metadynamics"
HILLS_FILENAME = "hills.npz"
FES_FILENAME = "fes.tsv"

LOG = logging.getLogger("neomd.methods.metadynamics")


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
    """Outcome of one metadynamics run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count
    fgroup: int  # force-group id of the installed bias
    n_hills: int  # hills deposited (incl. replayed ones after a resume)
    fes_sum: float  # sum of the free-energy grid (kJ/mol) — a drift sentinel
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# grid standardization — openmm BiasVariable._standardize for md_unit_system
# (through THE shared port table: port.to_canonical / port.cv_is_angular,
# the same conversion the fake kernel and the colvar tapes use)
# ---------------------------------------------------------------------------


def _is_angular(cv: CVIR) -> bool:
    """Angular CVs are declared in degrees (dihedral, angle)."""
    return cv_is_angular(cv)


def _standardize(value: float, cv: CVIR) -> float:
    """degree -> radian for angular CVs, nm as-is — openmm's md unit system.

    openmm's ``BiasVariable._standardize`` converts declared values into the
    md unit system, whose angle unit is the RADIAN (``(1*degree).
    value_in_unit_system(md_unit_system)`` == pi/180 — the exact factor in
    port.CANONICAL_FACTORS).  The kernel-side table limits and the values
    returned by ``bias_ops().cv_values()`` then agree exactly.
    """
    if _is_angular(cv):
        return to_canonical(value, "deg")
    return float(value)


def _grid_unit(cv: CVIR) -> str:
    """Kernel unit of one CV's deposition grid (the fes.tsv column header):
    radians for angular CVs, nm for nanometric ones (distances, RMSD, path
    z), '' for the dimensionless CVs (coordination, path s)."""
    if _is_angular(cv):
        return "rad"
    if cv.kind == "CustomNonbondedForce":
        return ""  # dimensionless coordination number
    if cv.kind == "PathCV" and cv.expression == "s":
        return ""  # dimensionless path progress
    return "nm"


def _make_evaluator(entry, cv: CVIR) -> Callable:
    """(positions, masses) -> float closure around one colvars entry.

    ColvarProbe evaluators report in the CV's NATURAL unit (degrees for
    dihedrals — the colvars.py evaluate convention, human-facing record);
    the metadynamics grid itself is kernel-standardized (radians).
    """

    def evaluate(positions, masses):
        return entry.evaluate(positions, masses, cv)

    return evaluate


# ---------------------------------------------------------------------------
# the well-tempered height (v1's openmm Quantity arithmetic, exact float port)
# ---------------------------------------------------------------------------


def _tempered_height(height: float, energy: float, delta_t: float) -> float:
    """v1's well-tempered hill height, bit-exact float port of the openmm
    Quantity arithmetic::

        height = self.height * np.exp(
            -energy / (unit.MOLAR_GAS_CONSTANT_R * self._deltaT))

    with ``energy`` a kJ/mol Quantity, ``self._deltaT`` a kelvin Quantity and
    ``self.height`` a kJ/mol Quantity.  openmm's Quantity operators
    (``openmm/unit/quantity.py``) evaluate that expression as a very specific
    float sequence, empirically characterized over 2e5 (T, biasFactor, height,
    energy) draws (energies up to 200 kJ/mol; 0 mismatches):

    * ``R * deltaT``   — ``Quantity * Quantity`` delegates to
      ``(Quantity * scalar) * unit``; the scalar multiply PRE-multiplies, so
      the denominator value is ``deltaT * 8.31446261815324`` (J/mol);
    * ``-energy / den`` — ``Quantity / Quantity`` delegates to
      ``(Quantity / scalar) / unit``; division by a scalar is multiplication
      by ``pow(den, -1.0)`` (bitwise ``1.0/den`` here — glibc pow is
      correctly rounded), and dividing out the J/mol unit then scales the
      kJ/mol numerator by the kJ->J factor 1000 with ANOTHER pre-multiply:
      ``1000.0 * ((1.0/den) * (-energy))``;
    * ``height * np.exp(arg)`` — the scalar pre-multiply again:
      ``np.exp(arg) * height`` (``np.exp`` and ``math.exp`` agreed bitwise on
      every probed argument);
    * ``value_in_unit(kilojoules_per_mole)`` on an already-kJ/mol Quantity is
      an identity conversion (factor 1.0, no multiply).

    The naive kJ-space form ``height * exp(-energy / (R_kJ * deltaT))``
    differs from this sequence by 1 ulp for ~75% of draws — enough to break
    bit-exact energy parity against the v1 tapes (tests/v2/test_parity_full).
    """
    denominator = delta_t * _R_J_MOL_K
    return math.exp(1000.0 * ((1.0 / denominator) * (-energy))) * height


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


class MetadynamicsRun:
    """One well-tempered metadynamics execution over a kernel.

    The driver's loop stops exactly at ``plan.steps``, so hills land on
    ``floor(steps/frequency)`` multiples; a resumed run's bias matrix is
    bit-identical to a straight run's (asserted in tests/v2 on the fake
    tier).

    Construct directly for artifact access (``get_free_energy`` /
    ``write_fes`` survive the run).  ``prepare()`` is the registry entry
    drive() dispatches; ``run()`` is the direct-construction convenience
    (prepare + the driver's method loop) returning the
    :class:`MethodResult`.
    """

    def __init__(self, kernel, plan, sink=None, logger=None):
        self.kernel = kernel
        self.plan = plan
        self.sink = sink
        self.log = LOG if logger is None else logger

        # -- meta_set -------------------------------------------------------
        meta = dict(getattr(plan, "meta_set", None) or {})
        missing = [key for key in ("biasFactor", "height", "frequency")
                   if key not in meta]
        if missing:
            raise ValueError(
                f"meta_set requires {', '.join(missing)} "
                f"(got keys {sorted(meta)})")
        if float(meta["biasFactor"]) <= 1.0:
            raise ValueError("biasFactor should be > 1.0")
        self.bias_factor = float(meta["biasFactor"])
        self.height = float(meta["height"])  # kJ/mol
        self.frequency = int(meta["frequency"])
        if self.frequency < 1:
            raise ValueError(
                f"meta_set.frequency must be >= 1, got {self.frequency}")
        update_context = meta.get("update_context_frequency")
        self.update_context_frequency = (None if update_context is None
                                         else int(update_context))
        self.temperature = float(getattr(plan, "temperature", 298.0))
        self.deltaT = self.temperature * (self.bias_factor - 1.0)

        # -- collective variables through the cv registry ------------------
        colvar_cfg = dict(getattr(plan, "colvars", None) or {})
        if not colvar_cfg:
            raise ValueError("metadynamics requires plan.colvars (1-3 entries)")

        import neomd.colvars  # noqa: F401  (import = cv registration)
        from neomd import registry

        self.cvs: list[tuple[str, CVIR, object]] = []  # (name, CVIR, entry)
        self.grids: list[GridSpec] = []  # kernel-standardized, config order
        self._scaled_variance: list[float] = []
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
            self.cvs.append((name, cv, entry))
            self.grids.append(gspec)
            # openmm BiasVariable scaled variance, verbatim:
            self._scaled_variance.append(
                (gspec.width / (gspec.maximum - gspec.minimum)) ** 2)

        # total bias over the REVERSED-axis grid
        self._shape = tuple(g.bins for g in reversed(self.grids))
        self._total_bias = np.zeros(self._shape, dtype=np.float64)
        self._hills_steps: list[int] = []
        self._hills_positions: list[list[float]] = []
        self._hills_heights: list[float] = []
        self._last_update_context_step = 0
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
            initial=self._total_bias.flatten().copy(),
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
                f"({', '.join(needed)}); metadynamics cannot run on it")
        self._ops = ops

        # resume through the single owner (neomd.resume.plan_resume): the
        # kernel restore + tape trimming happen there, AFTER install_bias
        # (forcing the openmm Context earlier would flip bias installs onto
        # the reinitialize path — see kernel/openmm.py).  What stays here is
        # the method's own physics: replaying the (already-trimmed) ledger.
        from neomd.resume import plan_resume

        resume_plan = plan_resume(self.plan, self.kernel, self.sink)
        if resume_plan is not None:
            self._replay_ledger(resume_plan)

        tapes: dict = {}
        if self.sink is not None:
            from neomd.probes import ColvarProbe

            tapes["colvar.tsv"] = ColvarProbe(
                self.sink,
                interval=self.frequency,
                cvs=[{"label": name, "evaluate": _make_evaluator(entry, cv)}
                     for name, cv, entry in self.cvs],
                masses=self.kernel.masses,
                append=resume_plan is not None
                and "colvar.tsv" in resume_plan.trims,
            )
        return PreparedMethod(
            on_step=self._deposit,
            on_step_interval=self.frequency,
            fgroups={LABEL: [self.fgroup]},
            resume_plan=resume_plan,
            tapes=tapes,
            progress=lambda step: {HILLS_FILENAME: int(step)},
            finish=self._finish,
        )

    def _finish(self, result) -> MethodResult:
        """End-of-run artifacts + the result drive() records."""
        from neomd.driver import CHECKPOINT_FILENAME

        # end-of-run artifacts: ledger + checkpoint + fes
        self._save_hills()
        if self.sink is not None:
            self.sink.write_bytes(CHECKPOINT_FILENAME, self.kernel.snapshot())
            try:
                self.write_fes(self.sink.path(FES_FILENAME))
            except NotImplementedError:
                pass  # filesystem-less sink (MemorySink): fes stays in memory
        return MethodResult(
            steps_done=result.steps_done,
            fgroup=self.fgroup,
            n_hills=len(self._hills_steps),
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

    # -- resume ---------------------------------------------------------------

    def _replay_ledger(self, resume_plan) -> None:
        """Replay the hills ledger into the bias matrix and push it.

        The ledger stores the hills themselves; the bias matrix is REBUILT
        by replaying them through the deposition math (deterministic), and
        one ``update_table`` pushes the rebuilt matrix to the kernel.

        The kernel restore and the ledger TRIM (hills deposited past the
        checkpoint step are dropped before replay) belong to
        :func:`neomd.resume.plan_resume` — the single resume owner; this
        method only replays what the trimmed ledger holds.
        """
        if self.sink is None:
            raise ValueError(
                f"continue_md needs a sink to load {HILLS_FILENAME} from")
        try:
            data = self.sink.read_bytes(HILLS_FILENAME)
        except (KeyError, FileNotFoundError):
            path = None
            try:
                path = self.sink.path(HILLS_FILENAME)
            except NotImplementedError:
                pass
            raise FileNotFoundError(
                f"cannot continue metadynamics: {HILLS_FILENAME} not found"
                + (f" at {path}" if path else "")) from None
        with np.load(io.BytesIO(data)) as loaded:
            steps = np.asarray(loaded["steps"], dtype=np.int64)
            heights = np.asarray(loaded["heights"], dtype=np.float64)
            positions = np.asarray(loaded["positions"], dtype=np.float64)
        ncv = len(self.grids)
        positions = positions.reshape(len(steps), ncv)
        for step, position, height in zip(steps.tolist(),
                                          positions.tolist(),
                                          heights.tolist()):
            self._add_gaussian(position, height)  # replay, no context push
            self._hills_steps.append(step)
            self._hills_positions.append(position)
            self._hills_heights.append(height)
        self._ops.update_table(LABEL, self._total_bias.flatten())
        self.log.info("Load bias FILE:%s (%d hills replayed)",
                      HILLS_FILENAME, len(steps))

    # -- the deposition cycle -------------------------------------------------

    def _deposit(self, step: int, view) -> None:
        """One hill deposit (the verbatim v1 cycle body)."""
        position = self._ops.cv_values(LABEL)
        energy = self._ops.bias_energy(LABEL)  # kJ/mol, the force group alone
        height = _tempered_height(self.height, energy, self.deltaT)
        self.log.info(
            "Starting Metadynamics cycle %d at time %s (step %d, "
            "bias energy %.6f kJ/mol)",
            len(self._hills_steps) + 1, datetime.datetime.now(), step, energy)
        self._add_gaussian(position, height)
        self._hills_steps.append(int(step))
        self._hills_positions.append([float(v) for v in position])
        self._hills_heights.append(float(height))
        self._save_hills()
        if self._update_context_check(step):
            self._ops.update_table(LABEL, self._total_bias.flatten())

    def _add_gaussian(self, position, height) -> None:
        """v1 ``_addGaussian`` (== openmm app/metadynamics.py), verbatim math.

        Per-axis Gaussians on the ``linspace(0, 1, bins)`` grid (INCLUSIVE
        of 1.0, so grid point ``i`` sits at
        ``minimum + i*(maximum-minimum)/(bins-1)``); the periodic distance
        handling including the ``dist[-1] = dist[0]`` seam; the
        reversed-axis outer product
        ``reduce(np.multiply.outer, reversed(axisGaussians))``; heights
        accumulate in kJ/mol.
        """
        axis_gaussians = []
        for i, grid in enumerate(self.grids):
            x = (position[i] - grid.minimum) / (grid.maximum - grid.minimum)
            if grid.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1.0, num=grid.bins) - x)
            if grid.periodic:
                dist = np.min(np.array([dist, np.abs(dist - 1)]), axis=0)
                dist[-1] = dist[0]
            axis_gaussians.append(
                np.exp(-0.5 * dist * dist / self._scaled_variance[i]))
        if len(self.grids) == 1:
            gaussian = axis_gaussians[0]
        else:
            gaussian = reduce(np.multiply.outer, reversed(axis_gaussians))
        self._total_bias += height * gaussian

    def _update_context_check(self, step: int) -> bool:
        """Whether to push the bias table to the kernel at ``step``.

        ``update_context_frequency is None`` -> push every hill; otherwise
        push only when more than ``update_context_frequency`` steps have
        elapsed since the last push.  The bias matrix always accumulates;
        only the kernel/context push is throttled.
        """
        if self.update_context_frequency is None:
            return True
        if step - self._last_update_context_step > self.update_context_frequency:
            self._last_update_context_step = step
            return True
        return False

    # -- artifacts -----------------------------------------------------------------

    def _save_hills(self) -> None:
        """hills.npz — the deposit ledger ``{steps, positions (n, ncv),
        heights (n,)}`` in kernel CV units (nm / radian).

        Method STATE, written by the deposit hook itself (NOT a
        switch-gated tape: a probe fires BEFORE ``on_step`` at a shared
        boundary, so a probe-written ledger would lag one deposit and
        break bit-exact resume).  Resume replays it through the
        deposition math (see ``_replay_ledger``).
        """
        if self.sink is None:
            return
        steps = np.asarray(self._hills_steps, dtype=np.int64)
        positions = np.asarray(self._hills_positions, dtype=np.float64)
        if positions.size == 0:
            positions = positions.reshape(0, len(self.grids))
        heights = np.asarray(self._hills_heights, dtype=np.float64)
        buffer = io.BytesIO()
        np.savez(buffer, steps=steps, positions=positions, heights=heights)
        self.sink.write_bytes(HILLS_FILENAME, buffer.getvalue())

    def get_free_energy(self) -> np.ndarray:
        """``-((T+deltaT)/deltaT) * total_bias``, kJ/mol.

        Shape is the reversed-axis grid convention (``tuple(bins for grid in
        reversed(grids))``), values at the ``linspace(0, 1, bins)`` grid
        points of each axis.
        """
        return (-((self.temperature + self.deltaT) / self.deltaT)
                * self._total_bias)

    def write_fes(self, path) -> None:
        """Write ``fes.tsv``: one row per grid point.

        Columns: each CV's coordinate in its kernel unit (nm / radian;
        dimensionless CVs — coordination, path s — carry no unit tag)
        followed by the free energy in kJ/mol.  Rows follow the bias array's
        C order (the reversed-axis convention: the FIRST configured CV
        varies fastest, the LAST slowest).
        """
        fes = self.get_free_energy()
        coords = [np.linspace(grid.minimum, grid.maximum, num=grid.bins)
                  for grid in self.grids]
        header = "# " + "\t".join(
            f"{name} [{unit}]" if (unit := _grid_unit(cv)) else name
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
                    "cv registry's keys (e.g. grp1_idx/grp2_idx, min_cv_nm, "
                    "max_cv_nm, biasWidth_nm, bins); 1-3 CVs"),
        "meta_set": ("mapping with biasFactor (> 1.0), height (kJ/mol), "
                     "frequency (steps between hills)"),
        "steps": "int, total steps (plan-level key)",
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "meta_set.update_context_frequency": (
            "int steps; None (default) pushes the bias table to the kernel "
            "on every hill, a number throttles the push like v1"),
        "continue_md": ("bool; restore output.ckpt and replay hills.npz from "
                        "the output directory before running"),
        "output.*": ("output_dir + state/trajectory/checkpoint intervals "
                     "(plan-level; the colvar recorder always fires on "
                     "meta_set.frequency)"),
    },
}


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'metadynamics'."""
    return MetadynamicsRun(kernel, plan, sink=sink, logger=logger).prepare()


register("method", "metadynamics", Method(schema=SCHEMA, prepare=_prepare))
