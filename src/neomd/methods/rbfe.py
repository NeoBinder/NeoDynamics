"""RBFE λ window — a method knowledge triple (issue #8 / W3-a; ADR-0007).

ONE alchemical window: dynamics at a fixed λ plus the du tape (``du.tsv``,
:class:`neomd.probes.DuProbe`) that records, at every report interval, the
potential energy at EVERY ladder λ through the port's ParamEnergy
capability — the BAR/MBAR input artifact.  The λ LADDER (N such windows run
as sequential legs, each its own run dir) is owned by :mod:`neomd.rbfe`,
the runner-level orchestrator; this triple is what one window's plan
(``method: "rbfe"``) dispatches to through drive().

How a window's λ reaches the kernel (ADR-0003's two paths):

* **openmm** — the alchemical system (one per experiment, built at the
  prepare boundary by :mod:`neomd.alchemical` from openmmtools) exposes λ
  as Context global parameters; the window's values ride
  ``KernelSpec.global_parameters`` (plan ``alchemical.lambda_values`` →
  :func:`neomd.run.build_kernel_spec`) and are applied at Context
  creation.  Nothing is installed here.
* **fake** — the kernel has no nonbonded physics to alchemify, so the
  window plan carries ``alchemical.mock_bias`` (two atom groups, a force
  constant, an equilibrium distance): prepare() installs a λ-scaled
  distance bias ``lambda_alchemical*(k/2)(d - r0)^2`` whose parameter
  value is the window's λ.  The mock gives the du tape well-defined
  λ-dependent energies on the fake kernel (ADR-0003's "mock λ-偏置"
  decision) — orchestration/resume mechanics are what the fake tier
  proves, never the softcore physics (settled decision #9).

The boresch anchor (restraints.py) is wired by the WINDOW PLAN's ordinary
``restraint:`` section — drive() installs it like any restraint and the
restraint tape reports it; this method never touches restraint wiring.
Its energy is identical across λ, so it cancels exactly in every du
difference and in the BAR/MBAR estimators (per-sample constants).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping

from neomd.kernel.port import BiasIR, Param, ParamEnergy, provides
from neomd.registry import register

__all__ = [
    "Method",
    "RbfeMethodResult",
    "LABEL",
    "DU_FILENAME",
    "MOCK_PARAMETER",
]

LABEL = "rbfe"
DU_FILENAME = "du.tsv"
#: the mock bias's λ parameter name (the fake-kernel window's λ spelling)
MOCK_PARAMETER = "lambda_alchemical"

LOG = logging.getLogger("neomd.methods.rbfe")


@dataclass(frozen=True)
class Method:
    """One method knowledge triple: schema + prepare (registry kind "method").

    Same contract as metadynamics/smd: ``prepare(kernel=..., plan=...,
    sink=..., logger=...) -> neomd.driver.PreparedMethod``; the driver runs
    the loop and owns reporting (driver.run_prepared_method).
    """

    schema: dict
    prepare: Callable


@dataclass(frozen=True)
class RbfeMethodResult:
    """Outcome of one λ window (drive() appends it to RunOutcome.results)."""

    steps_done: int  # final absolute step count
    lambda_values: dict  # the window's λ ({parameter name: value})
    fgroups: list  # the mock bias's force group(s) ([] on the openmm path)
    du_last_step: int | None  # last step written to du.tsv (None: no tape)
    positions_sha256: str = ""  # sha256 of the final positions (float64 nm)


def _mock_bias(lambda_values: Mapping[str, float], mock: Mapping) -> BiasIR:
    """The fake-kernel λ-scaled distance bias (ADR-0003's mock decision).

    ``lambda_alchemical*(k/2)*(distance(g1,g2) - r0)^2`` — k and r0 are
    inlined into the expression (only λ is a live parameter), the window's
    λ is the parameter VALUE, so the dynamics run at the window's λ and
    ``energy_with_params`` can probe the neighboring λ values.
    """
    if MOCK_PARAMETER not in lambda_values:
        raise ValueError(
            f"alchemical.mock_bias windows key their λ as "
            f"{MOCK_PARAMETER!r} in lambda_values (got "
            f"{sorted(lambda_values)})")
    lam = float(lambda_values[MOCK_PARAMETER])
    k = float(mock["k_kj_mol_nm2"])
    r0 = float(mock["r0_nm"])
    energy = f"{MOCK_PARAMETER}*({k}/2)*(distance(g1,g2) - {r0})^2"
    return BiasIR(
        kind="CustomCentroidBondForce",
        energy=energy,
        params={MOCK_PARAMETER: Param(lam, "dimensionless")},
        groups=[[int(index) for index in _indices(mock["grp1_idx"])],
                [int(index) for index in _indices(mock["grp2_idx"])]],
        periodic=True,
        label="alchemical",
    )


def _indices(value) -> list[int]:
    """grp_idx value (int | list | comma-string) -> list[int]."""
    if isinstance(value, str):
        return [int(v) for v in value.split(",")]
    if isinstance(value, (int, float)):
        return [int(value)]
    return [int(v) for v in value]


def _prepare(kernel, plan, sink=None, logger=None):
    """Registry entry point — drive() calls this for method 'rbfe'."""
    from neomd.driver import PreparedMethod

    log = LOG if logger is None else logger
    alchemical = dict(getattr(plan, "alchemical", None) or {})
    lambda_values = dict(alchemical.get("lambda_values") or {})
    ladder = list(alchemical.get("ladder") or [])
    if not lambda_values or not ladder:
        raise ValueError(
            "method 'rbfe' requires the alchemical section (lambda_values "
            "+ ladder); see methods/rbfe.py")

    if not provides(kernel, ParamEnergy):
        raise NotImplementedError(
            f"kernel {kernel.name!r} does not provide the ParamEnergy "
            f"capability (energy_with_params); RBFE windows need cross-λ "
            f"energies for the du tape")

    # the mock bias: only when the plan asks for it (fake-kernel windows);
    # openmm windows ride KernelSpec.global_parameters instead
    fgroups: list = []
    mock = alchemical.get("mock_bias")
    if mock is not None:
        bias = _mock_bias(lambda_values, mock)
        fgroups = [kernel.install_bias(bias)]
        log.info("mock alchemical bias installed as force group %s "
                 "(lambda=%s)", fgroups, lambda_values)

    # resume through the single owner, AFTER install_bias (the contract)
    from neomd.resume import plan_resume

    resume_plan = plan_resume(plan, kernel, sink)

    tapes: dict = {}
    du_probe = None
    if sink is not None:
        from neomd.probes import DuProbe

        interval = int(getattr(plan, "report_interval", 0) or 0)
        if interval <= 0:
            raise ValueError(
                "method 'rbfe' needs output.report_interval > 0 (the du "
                "tape's cadence — it is the BAR/MBAR input artifact)")
        du_probe = DuProbe(
            sink,
            interval=interval,
            ladder=ladder,
            append=resume_plan is not None and DU_FILENAME in resume_plan.trims,
            resume_step=(resume_plan.trims.get(DU_FILENAME)
                         if resume_plan is not None else None),
        )
        tapes[DU_FILENAME] = du_probe
    return PreparedMethod(
        on_step=None,
        on_step_interval=1,
        fgroups={LABEL: fgroups},
        resume_plan=resume_plan,
        tapes=tapes,
        finish=lambda result: _finish(result, kernel, plan, sink, du_probe,
                                      lambda_values, fgroups),
    )


def _finish(result, kernel, plan, sink, du_probe, lambda_values, fgroups
            ) -> RbfeMethodResult:
    from neomd.driver import CHECKPOINT_FILENAME

    if sink is not None:
        sink.write_bytes(CHECKPOINT_FILENAME, kernel.snapshot())
    return RbfeMethodResult(
        steps_done=result.steps_done,
        lambda_values=dict(lambda_values),
        fgroups=list(fgroups),
        du_last_step=None if du_probe is None else du_probe.last_step,
        positions_sha256=result.positions_sha256,
    )


# ---------------------------------------------------------------------------
# schema + registration (the knowledge triple)
# ---------------------------------------------------------------------------

SCHEMA = {
    "required": {
        "alchemical": ("mapping with lambda_values (THIS window's λ, "
                       "{Context parameter: value in [0,1]}) and ladder "
                       "(every window's lambda_values, in ladder order — "
                       "the du tape's column vocabulary); optional "
                       "mock_bias {grp1_idx, grp2_idx, k_kj_mol_nm2, r0_nm} "
                       "for fake-kernel windows"),
        "steps": "int, total steps of the window leg (plan-level key)",
        "temperature": "number, kelvin (plan-level key)",
    },
    "optional": {
        "restraint": ("the anchoring section — boresch over 3+3 anchor "
                      "atoms keeps the decoupled ligand oriented"),
        "continue_md": ("bool; resume this window from its output.ckpt "
                        "(the ladder orchestrator sets it when a window "
                        "was interrupted)"),
        "output.*": ("output_dir + report_interval (the du tape cadence) + "
                     "state/trajectory/checkpoint intervals"),
    },
}


register("method", "rbfe", Method(schema=SCHEMA, prepare=_prepare))
