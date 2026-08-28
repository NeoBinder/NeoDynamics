"""Well-tempered metadynamics — a method knowledge triple (plan §5 item 2.2).

Verbatim port of v1 ``src/neomd/metadynamics/engine.py`` (``MetadynamicsEngine``,
itself derived from openmm ``app/metadynamics.py``) onto the v2 seams: the
bias is a ``BiasIR(kind="CustomCVTableForce")`` installed through
``kernel.install_bias`` (compiled by the kernel exactly like v1's
``prepare_metadynamics_bias``), the deposition cycle rides the driver's
``on_step`` hook (``on_step_interval = meta_set.frequency`` — the driver's
boundary arithmetic replaces v1's hand-rolled ``while stepsToGo > 0`` chunking,
landing on the exact same multiples), and live bias access goes through
``kernel.bias_ops()`` (cv_values / bias_energy / update_table).

Physics ported VERBATIM (discipline §8 #3 — the numbers, not just the shape):

* hill height (v1 ``run_md`` lines 296-302):
  ``height * exp(-E_bias / (R * deltaT))`` with ``deltaT = T*(biasFactor-1)``
  and ``E_bias`` the force-group energy of the metadynamics force alone.
  The tempering argument reproduces v1's openmm Quantity arithmetic
  BIT-EXACTLY in pure floats (:func:`_tempered_height` below; openmm is
  never imported here, methods stay kernel-agnostic).
* ``_addGaussian`` (v1 lines 191-217 == openmm metadynamics.py 193-219):
  per-axis Gaussians on the ``linspace(0, 1, bins)`` grid (INCLUSIVE of 1.0,
  so grid point ``i`` sits at ``minimum + i*(maximum-minimum)/(bins-1)``),
  v1's periodic distance handling including the ``dist[-1] = dist[0]`` seam,
  the reversed-axis outer product
  ``reduce(np.multiply.outer, reversed(axisGaussians))``, and the
  height-accumulation in kJ/mol.
* ``_scaledVariance`` — the port source is openmm's ``BiasVariable``
  (``openmm/app/metadynamics.py`` line 305):
  ``self._scaledVariance = (self.biasWidth/(self.maxValue-self.minValue))**2``
  — the width normalized by the grid RANGE (dimensionless, so the unit
  cancels).  ``BiasVariable.__init__`` also *standardizes* min/max/width via
  ``value_in_unit_system(md_unit_system)``; openmm's md unit system is
  radian-based (verified: ``(1*degree).value_in_unit_system(md_unit_system)``
  == pi/180), so angular CV grids declared in degrees become radians before
  they reach the kernel — matching what openmm's own
  ``getCollectiveVariableValues`` returns for torsion CVs (verified: radians).
  Distance grids are nanometers on both kernels already.
* ``update_context_check`` (v1 lines 160-173): ``update_context_frequency is
  None`` -> push the table every hill; otherwise push only when more than
  ``update_context_frequency`` steps elapsed since the last push.  The bias
  matrix always accumulates; only the kernel/context push is throttled.

Deliberate deviations (documented, none touches the physics):

* artifacts are BRAND NEW (plan R3-Q3): ``colvar.tsv`` (ColvarProbe, natural
  CV units — degrees for dihedrals) and ``hills.npz`` — the hill LEDGER
  ``{steps, positions (n, ncv), heights (n,)}`` in kernel CV units — replace
  v1's ``COLVAR.npy`` + ``bias_last.npy``.  Old consumers (gethill/hills_ana)
  break on flip day, acknowledged.
* resume (v1 ``continue_metadynamics``): instead of persisting the whole bias
  matrix, the ledger is REPLAYED through the deposition math
  (deterministic — same hills in, same matrix out), then pushed to the kernel
  once, exactly where v1 called ``updateParametersInContext``.  A resumed
  run's hills are bit-identical to a straight run's (the §6 meta-resume row,
  asserted in tests/v2 on the fake tier).
* v1's cycle loop ran ``ceil(steps/frequency)`` cycles (overshooting
  ``steps`` when the two are not commensurate); the v2 driver contract stops
  exactly at ``plan.steps``, so hills land on ``floor(steps/frequency)``
  multiples.  For commensurate runs (the golden scenario 2000/100, all tests)
  the behavior is identical.
"""

from __future__ import annotations

import datetime
import io
import logging
import math
import os
from dataclasses import dataclass
from functools import reduce
from typing import Callable

import numpy as np

from neomd.kernel.port import BiasIR, CVIR, GridSpec, TableSpec
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
    """One method knowledge triple: schema + run (registry kind "method").

    ``run`` has the drive() dispatch signature
    ``run(kernel=..., plan=..., sink=..., logger=...) -> MethodResult``.
    """

    schema: dict
    run: Callable


@dataclass(frozen=True)
class MethodResult:
    """Outcome of one metadynamics run (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count
    fgroup: int  # force-group id of the installed bias (v1 max of free groups)
    n_hills: int  # hills deposited (incl. replayed ones after a resume)
    fes_sum: float  # sum of the free-energy grid (kJ/mol) — a drift sentinel
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


# ---------------------------------------------------------------------------
# grid standardization — openmm BiasVariable._standardize for md_unit_system
# ---------------------------------------------------------------------------


def _is_angular(cv: CVIR) -> bool:
    """Angular CVs are declared in degrees by v1 configs (dihedral, angle)."""
    if cv.kind == "CustomTorsionForce":
        return True
    return "angle(" in cv.expression.replace(" ", "")


def _standardize(value: float, cv: CVIR) -> float:
    """degree -> radian for angular CVs, nm as-is — openmm's md unit system.

    Port of ``BiasVariable._standardize(minValue*degree, ...)``: v1 handed the
    engine degree Quantities and BiasVariable converted them into the md unit
    system, whose angle unit is the RADIAN (verified against
    ``unit.md_unit_system``).  The kernel-side table limits and the values
    returned by ``bias_ops().cv_values()`` then agree exactly as in v1.
    """
    if _is_angular(cv):
        return math.radians(float(value))
    return float(value)


def _grid_unit(cv: CVIR) -> str:
    return "rad" if _is_angular(cv) else "nm"


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
    Quantity arithmetic in ``engine.py:299-301``::

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

    Construct directly for artifact access (``get_free_energy`` /
    ``write_fes`` survive the run); ``run()`` returns the
    :class:`MethodResult` that drive() records.
    """

    def __init__(self, kernel, plan, sink=None, logger=None):
        self.kernel = kernel
        self.plan = plan
        self.sink = sink
        self.log = LOG if logger is None else logger

        # -- meta_set (v1 engine __init__) ---------------------------------
        meta = dict(getattr(plan, "meta_set", None) or {})
        missing = [key for key in ("biasFactor", "height", "frequency")
                   if key not in meta]
        if missing:
            raise ValueError(
                f"meta_set requires {', '.join(missing)} "
                f"(got keys {sorted(meta)})")
        if float(meta["biasFactor"]) <= 1.0:
            raise ValueError("biasFactor should be > 1.0")  # v1 message
        self.bias_factor = float(meta["biasFactor"])
        self.height = float(meta["height"])  # kJ/mol (v1 kilojoules_per_mole)
        self.frequency = int(meta["frequency"])
        if self.frequency < 1:
            raise ValueError(
                f"meta_set.frequency must be >= 1, got {self.frequency}")
        update_context = meta.get("update_context_frequency")
        self.update_context_frequency = (None if update_context is None
                                         else int(update_context))
        self.temperature = float(getattr(plan, "temperature", 298.0))
        # v1: self._deltaT = temperature * (biasFactor - 1)
        self.deltaT = self.temperature * (self.bias_factor - 1.0)

        # -- collective variables through the cv registry (v1 generate_colvar)
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
            # openmm BiasVariable line 305, verbatim:
            self._scaled_variance.append(
                (gspec.width / (gspec.maximum - gspec.minimum)) ** 2)

        # v1 prepare_metadynamics_bias: total bias over the REVERSED-axis grid
        self._shape = tuple(g.bins for g in reversed(self.grids))
        self._total_bias = np.zeros(self._shape, dtype=np.float64)
        self._hills_steps: list[int] = []
        self._hills_positions: list[list[float]] = []
        self._hills_heights: list[float] = []
        # v1 engine __init__: self.last_update_context_step = 0
        self._last_update_context_step = 0
        self.fgroup: int | None = None

    # -- entry point --------------------------------------------------------

    def run(self) -> MethodResult:
        """Install the table bias, (optionally) resume, and run the loop."""
        from neomd.driver import CHECKPOINT_FILENAME, _default_probes, run_md

        var_names = ["cv%d" % i for i in range(len(self.grids))]
        table = TableSpec(
            cvs=[cv for _, cv, _ in self.cvs],
            grids=list(self.grids),
            initial=self._total_bias.flatten().copy(),
            label=LABEL,
        )
        bias = BiasIR(
            kind="CustomCVTableForce",
            energy="table(%s)" % ", ".join(var_names),  # v1 expression
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

        resume = bool(getattr(self.plan, "continue_md", False))
        if resume:
            self._resume()

        # -- probes: the plan's defaults + the colvar recorder ---------------
        probes = _default_probes(self.plan, self.sink)
        if self.sink is not None:
            from neomd.probes import ColvarProbe

            probes.append(ColvarProbe(
                self.sink,
                interval=self.frequency,
                cvs=[{"label": name, "evaluate": _make_evaluator(entry, cv)}
                     for name, cv, entry in self.cvs],
                masses=getattr(self.kernel, "masses", None),
                append=resume,  # v1 continue_md appended to COLVAR
            ))

        result = run_md(self.kernel, self.plan, probes,
                        on_step=self._deposit,
                        on_step_interval=self.frequency,
                        logger=self.log,
                        sink=self.sink)  # last.pdbx + last.ckpt (v1 save_last)

        # v1 save_last: bias + colvar + checkpoint at run end
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

    # -- resume (v1 continue_metadynamics) ------------------------------------

    def _resume(self) -> None:
        """Restore kernel state, replay the hills ledger, push the rebuilt bias.

        v1 loaded ``bias_last.npy`` (the full matrix) + ``COLVAR.npy``; the v2
        ledger stores the hills themselves and the bias matrix is REBUILT by
        replaying them through the deposition math — deterministic, and
        ``update_table`` replaces v1's unconditional
        ``setFunctionParameters + updateParametersInContext`` pair.
        """
        # Kernel state: kernels driven through drive() already restored via
        # KernelSpec.resume (the openmm Context path).  A kernel still at
        # step 0 restores here from the derived checkpoint path — this covers
        # the fake kernel (which does not read spec.resume) exactly like v1's
        # _create_simulation resume branch covered every engine.
        if self.kernel.current_step == 0:
            checkpoint = getattr(self.plan, "checkpoint", None)
            if not checkpoint:
                raise ValueError(
                    "continue_md is true but no checkpoint was derived "
                    "(input_files.checkpoint / output.output_dir)")
            with open(checkpoint, "rb") as handle:
                self.kernel.restore(handle.read())
            self.log.info("Load checkpoint FILE:%s (step %d)",
                          checkpoint, self.kernel.current_step)

        if self.sink is None:
            raise ValueError(
                f"continue_md needs a sink to load {HILLS_FILENAME} from")
        try:
            path = self.sink.path(HILLS_FILENAME)
        except NotImplementedError as error:
            raise ValueError(
                f"continue_md needs a filesystem sink to load "
                f"{HILLS_FILENAME}") from error
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"cannot continue metadynamics: {path} not found")
        with np.load(path) as data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            heights = np.asarray(data["heights"], dtype=np.float64)
            positions = np.asarray(data["positions"], dtype=np.float64)
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
        self.log.info("Load bias FILE:%s (%d hills replayed)", path, len(steps))

    # -- the deposition cycle (v1 run_md inner body) ---------------------------

    def _deposit(self, step: int, view) -> None:
        """One hill, the verbatim v1 cycle body (engine.py lines 292-302)."""
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
        # v1 saved its bias + COLVAR every cycle; the ledger replaces both
        self._save_hills()
        if self._update_context_check(step):
            self._ops.update_table(LABEL, self._total_bias.flatten())

    def _add_gaussian(self, position, height) -> None:
        """v1 ``_addGaussian`` (== openmm app/metadynamics.py), verbatim math."""
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
        # v1 converts the height Quantity to kJ/mol right here — same float
        self._total_bias += height * gaussian

    def _update_context_check(self, step: int) -> bool:
        """v1 ``update_context_check`` (engine.py lines 160-173)."""
        if self.update_context_frequency is None:
            return True
        if step - self._last_update_context_step > self.update_context_frequency:
            self._last_update_context_step = step
            return True
        return False

    # -- artifacts -----------------------------------------------------------------

    def _save_hills(self) -> None:
        """hills.npz — the NEW ledger (plan R3-Q3): steps, positions, heights."""
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
        """v1 ``get_free_energy``: ``-((T+deltaT)/deltaT) * totalBias`` kJ/mol.

        Shape is the reversed-axis grid convention (``tuple(bins for grid in
        reversed(grids))``), values at the ``linspace(0, 1, bins)`` grid
        points of each axis.
        """
        return (-((self.temperature + self.deltaT) / self.deltaT)
                * self._total_bias)

    def write_fes(self, path) -> None:
        """Write ``fes.tsv`` (new format): one row per grid point.

        Columns: each CV's coordinate in its kernel unit (nm / radian — the
        deposition grid's own units) followed by the free energy in kJ/mol.
        Rows follow the bias array's C order (v1's reversed-axis convention:
        the FIRST configured CV varies fastest, the LAST slowest).
        """
        fes = self.get_free_energy()
        coords = [np.linspace(grid.minimum, grid.maximum, num=grid.bins)
                  for grid in self.grids]
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


def _run(kernel, plan, sink=None, logger=None) -> MethodResult:
    """Registry entry point — drive() calls this for method 'metadynamics'."""
    return MetadynamicsRun(kernel, plan, sink=sink, logger=logger).run()


register("method", "metadynamics", Method(schema=SCHEMA, run=_run))
