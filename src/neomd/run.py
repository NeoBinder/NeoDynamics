"""
run — the C facade of neomd.

One entry point, three levels of disclosure: L0 ``md_run(dir)`` (zero-config
plan discovery), L1 ``md_run(dir, **top-level-overrides)`` (scalar knobs via
``plan.with_``; a whole section passed REPLACES it verbatim — no deep
merge), L2 ``md_run(plan_dict_or_Plan)``.  Round-trip law: all spellings
compile to an IDENTICAL Plan (identical fingerprints), because every level
funnels into the same ``Plan`` construction and the fingerprint is a pure
function of the raw config — pinned by tests/v2/test_run.py.  :func:`compile`
is the L2 companion; the ONE kernel-spec builder is
:func:`build_kernel_spec`.  This module never imports openmm (lazy adapter
bootstrap); plugins under the ``neomd`` entry-point group are scanned
before the Plan is built (ADR-0002).
"""

from __future__ import annotations

import fnmatch
import os
from typing import TYPE_CHECKING, Mapping

from .errors import ConfigKeyError
from .plan import KNOWN_KEYS, Plan, load_plan

if TYPE_CHECKING:  # annotation-only: the runtime import stays lazy in-body
    from .kernel.port import KernelSpec

__all__ = ["CompiledRun", "compile", "md_run", "build_kernel_spec",
           "PLAN_FILENAMES"]

#: L0 plan-file discovery: preferred names, in priority order (first hit wins).
PLAN_FILENAMES = ("neomd.yaml", "plan.yaml", "neomd.json", "plan.json")

#: L0 fallback patterns: when no preferred name exists, EXACTLY one file
#: matching these may be present (more -> ambiguity error).
PLAN_PATTERNS = ("*.yaml", "*.yml", "*.json")


# ---------------------------------------------------------------------------
# L0: plan-file discovery
# ---------------------------------------------------------------------------


def _discover_plan_file(directory: str) -> str:
    """Find the plan file inside *directory* (L0; see module docstring)."""
    for name in PLAN_FILENAMES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    found = sorted(
        {
            entry
            for pattern in PLAN_PATTERNS
            for entry in _match(directory, pattern)
        }
    )
    if not found:
        raise ConfigKeyError(
            f"no plan file found in {directory!r}; expected one of "
            f"{', '.join(PLAN_FILENAMES)}, or exactly one "
            f"{'/'.join(PLAN_PATTERNS)} file",
            known_keys=PLAN_FILENAMES,
        )
    if len(found) > 1:
        raise ConfigKeyError(
            f"ambiguous plan directory {directory!r}: found {len(found)} "
            f"candidate plan files ({', '.join(found)}); name the file "
            f"neomd.yaml or pass the file path directly",
            candidates=found,
            known_keys=PLAN_FILENAMES,
        )
    return os.path.join(directory, found[0])


def _match(directory: str, pattern: str) -> list[str]:
    try:
        entries = os.listdir(directory)
    except OSError as error:
        raise ConfigKeyError(
            f"cannot look for a plan file in {directory!r}: {error}",
            known_keys=PLAN_FILENAMES,
        ) from error
    return [entry for entry in entries if fnmatch.fnmatch(entry, pattern)]


# ---------------------------------------------------------------------------
# target -> Plan (the L0/L1/L2 funnel)
# ---------------------------------------------------------------------------


def _resolve_plan(target, overrides: Mapping) -> Plan:
    """Turn any md_run target (dir | file | dict | Plan) into a Plan.

    ``overrides`` are applied last via ``plan.with_`` (top-level keys only,
    re-validated from scratch) — that is the whole of L1.
    """
    if isinstance(target, Plan):
        plan = target
    elif isinstance(target, Mapping):
        plan = Plan.from_dict(dict(target))
    elif isinstance(target, (str, os.PathLike)):
        path = os.fspath(target)
        if os.path.isdir(path):
            plan = load_plan(_discover_plan_file(path))
        elif os.path.isfile(path):
            plan = load_plan(path)
        else:
            raise ConfigKeyError(
                f"md_run target {path!r} does not exist; expected a plan "
                f"file, a directory containing one "
                f"({', '.join(PLAN_FILENAMES)}), a plan dict, or a Plan",
                known_keys=PLAN_FILENAMES,
            )
    else:
        raise TypeError(
            f"md_run target must be a directory/file path, a plan dict, or "
            f"a Plan; got {type(target).__name__}")

    if overrides:
        unknown = sorted(key for key in overrides if key not in KNOWN_KEYS)
        if unknown:
            raise ConfigKeyError(
                f"unknown md_run override(s): {', '.join(repr(k) for k in unknown)} "
                f"(overrides are top-level plan keys; nested sections are "
                f"replaced wholesale — use the L2 dict form for surgery)",
                key=unknown[0],
                known_keys=KNOWN_KEYS,
            )
        plan = plan.with_(**overrides)
    return plan


# ---------------------------------------------------------------------------
# plugin loading (the facade's side of the plugin contract, ADR-0002)
# ---------------------------------------------------------------------------


def _scan_plugins() -> None:
    """Load installed plugin distributions before a Plan is built.

    ``plugins:`` sections are validated against the registry (unknown plugin
    names are collect-all errors), and plugin methods dispatch through it —
    so third-party entries must be registered before ``Plan`` construction.
    "Whoever starts a run scans once" (the gamd_drill contract): the facade
    entry points do it here; library callers building Plans themselves
    import or scan the plugin on their own.  With nothing installed under
    the ``neomd`` entry-point group this is a no-op metadata read.
    """
    from . import registry

    registry.scan_entry_points()


# ---------------------------------------------------------------------------
# Plan -> KernelSpec
# ---------------------------------------------------------------------------


def _particle_masses(system_modification) -> dict[int, float] | None:
    """``{particle index: mass}`` from raw ``system_modification`` entries.

    A mapping of ``{index: {"mass": value}}`` where every entry with a
    "mass" key sets that particle's mass (entries describing other
    modifications are ignored).  The list spelling plan.py also accepts
    (``[{"index": i, "mass": m}, ...]``) is normalized the same way.
    """
    if not system_modification:
        return None
    if isinstance(system_modification, Mapping):
        entries = system_modification.items()
    else:
        entries = (
            (entry.get("index"), entry)
            for entry in system_modification
            if isinstance(entry, Mapping)
        )
    masses: dict[int, float] = {}
    for index, info in entries:
        if isinstance(info, Mapping) and "mass" in info:
            masses[int(index)] = float(info["mass"])
    return masses or None


def _dummy_exceptions(system_modification) -> tuple[tuple[int, int], ...] | None:
    """Flattened ``(particle, partner)`` pairs from raw ``system_modification``
    entries: ``{index: {"dummy_atom_Nonbond_Exception": [partners...]}}``
    adds one zero-interaction NonbondedForce exception per pair.  Both the
    mapping and list spellings accepted by ``_particle_masses`` are
    normalized the same way.
    """
    if not system_modification:
        return None
    if isinstance(system_modification, Mapping):
        entries = system_modification.items()
    else:
        entries = (
            (entry.get("index"), entry)
            for entry in system_modification
            if isinstance(entry, Mapping)
        )
    pairs: list[tuple[int, int]] = []
    for index, info in entries:
        if not isinstance(info, Mapping):
            continue
        partners = info.get("dummy_atom_Nonbond_Exception")
        if partners is None:
            continue
        for partner in partners:
            pairs.append((int(index), int(partner)))
    return tuple(pairs) or None


def _global_parameters(alchemical) -> dict | None:
    """``{Context global parameter: value}`` from the raw ``alchemical``
    section's ``lambda_values`` (method ``"rbfe"``; ADR-0003/0007) — the
    per-window λ pushed through the KernelSpec seam the same way
    ``set_bias_param`` pushes ramps mid-run.  ``None`` when absent.
    """
    if not alchemical or not isinstance(alchemical, Mapping):
        return None
    values = alchemical.get("lambda_values")
    if not values:
        return None
    return {str(name): float(value) for name, value in values.items()}


def build_kernel_spec(plan: Plan, *, kind: str = "openmm",
                      platform: str = "cpu") -> KernelSpec:
    """Compile the plan into a :class:`~neomd.kernel.port.KernelSpec`.

    THE one spec builder: both ``compile()`` and direct ``drive()`` calls
    consume exactly this — there is no second, weaker spec path.

    * ``system_xml`` / ``topology_file`` come straight from ``input_files``;
    * the integrator dict is the RAW plan section (plus the defaults for
      the name/friction/dt keys the section may omit);
    * ``resume`` comes from the plan's DERIVED checkpoint/state (the
      ``continue_md`` resolution);
    * the barostat dict is the RAW section **augmented with the plan seed**
      (the adapter seeds the barostat from the plan seed, never its own);
      the openmm adapter takes pressure/frequency from it and defaults
      temperature to the plan temperature;
    * ``particle_masses`` from ``system_modification`` (see above).
    * ``ml_region`` (ADR-0004) is the RAW section verbatim — the
      openmm adapter assembles the mechanical embedding + NNP force from it
      BEFORE creating a Context (it must never reach system.xml: the NNP
      Force is not XML-serializable); ``residues`` selectors resolve there
      against the loaded complex topology; the fake kernel ignores it.
    """
    from .kernel.port import KernelSpec

    integrator = dict(getattr(plan, "integrator", None) or {})
    integrator.setdefault("integrator_name", "LangevinIntegrator")
    integrator.setdefault("friction_coeff", 1.0)
    integrator.setdefault("dt", 0.002)

    resume = None
    if getattr(plan, "continue_md", False):
        checkpoint = getattr(plan, "checkpoint", None)
        state = getattr(plan, "state", None)
        if checkpoint:
            resume = {"checkpoint": checkpoint}
        elif state:
            resume = {"state": state}

    barostat = getattr(plan, "barostat", None)
    if barostat:
        barostat = {**dict(barostat), "seed": int(plan.seed)}

    ml_region = getattr(plan, "ml_region", None)
    if ml_region:
        ml_region = dict(ml_region)  # off the frozen plan containers

    input_files = plan.input_files
    return KernelSpec(
        kind=kind,
        system_xml=str(input_files["system"]),
        topology_file=str(input_files["complex"]),
        integrator=integrator,
        temperature=float(plan.temperature),
        seed=int(plan.seed),
        platform=platform,
        resume=resume,
        barostat=barostat,
        particle_masses=_particle_masses(
            getattr(plan, "system_modification", None)),
        dummy_exceptions=_dummy_exceptions(
            getattr(plan, "system_modification", None)),
        ml_region=ml_region,
    )


# ---------------------------------------------------------------------------
# the facade
# ---------------------------------------------------------------------------


class CompiledRun:
    """A plan wired for execution: kernel + sink + driver.

    Built by :func:`compile`; holds the frozen :attr:`plan`, the live
    :attr:`kernel` (constructed eagerly — creating an openmm kernel builds
    the System and Context), and the :attr:`sink` on ``plan.output_dir``.
    :meth:`run` delegates to :func:`neomd.driver.drive`, which owns restraints,
    the method dispatch, default probes and the manifest.
    """

    __slots__ = ("plan", "kernel", "sink", "logger")

    def __init__(self, plan: Plan, kernel, sink, logger=None):
        self.plan = plan
        self.kernel = kernel
        self.sink = sink
        self.logger = logger

    def run(self):
        """Execute the plan through the driver; returns its ``RunOutcome``."""
        from .driver import drive

        # The factory closes over the compiled kernel: construction is
        # compile()'s job, orchestration is drive()'s (drive only sees the
        # kernel it is handed through this seam).
        return drive(
            self.plan,
            lambda spec: self.kernel,
            sink=self.sink,
            logger=self.logger,
        )

    def __repr__(self) -> str:
        return (f"CompiledRun(plan={self.plan!r}, kernel={self.kernel.name!r}, "
                f"sink={type(self.sink).__name__})")


def compile(plan_or_dict, *, kernel: str = "openmm", platform: str = "cpu",
            logger=None) -> CompiledRun:
    """
    Plan -> kernel + sink + driver wiring (the L2 companion of md_run).

    ``plan_or_dict``: a :class:`~neomd.plan.Plan` or the plan dict (validated
    and frozen; a dict triggers the plugin entry-point scan first — see
    ADR-0002 — so installed plugin sections validate and dispatch).
    ``platform`` passes through to the KernelSpec (default ``"cpu"``,
    matching the CI parity environment).  Raises
    :class:`NotImplementedError` for ``kernel="fake"``: the fake kernel is
    built from an in-memory ``SystemData``, not from plan input files — for
    fake-kernel runs build the kernel yourself and call
    ``neomd.driver.drive(plan, kernel_factory=...)``.
    """
    if isinstance(plan_or_dict, Plan):
        plan = plan_or_dict
    else:
        _scan_plugins()
        plan = Plan.from_dict(dict(plan_or_dict))

    if kernel == "fake":
        raise NotImplementedError(
            "compile(kernel='fake') is not supported: the fake kernel is "
            "built from an in-memory SystemData, not from plan input files. "
            "Build a FakeKernel yourself and run it with "
            "neomd.driver.drive(plan, kernel_factory=...) "
            "(see tests/v2/test_driver.py).")

    from .kernel._bootstrap import ensure_adapters
    from .kernel.port import KernelFactory
    from .sinks import LocalDirSink

    ensure_adapters()
    created = KernelFactory.create(build_kernel_spec(plan, kind=kernel,
                                                     platform=platform))
    return CompiledRun(plan, created, LocalDirSink(plan.output_dir), logger)


def md_run(target, *, platform: str = "cpu", kernel: str = "openmm",
           logger=None, **overrides):
    """Run an experiment; the entry point (see module docstring).

    Parameters
    ----------
    target:
        ``"dir"`` (L0/L1: a plan file is discovered inside) or a plan file
        path, or the plan dict / a :class:`Plan` (L2).
    platform:
        openmm platform passed through to the kernel (default ``"cpu"``).
    **overrides:
        L1: top-level plan keys replaced via ``plan.with_`` (unknown keys
        raise :class:`~neomd.errors.ConfigKeyError` with a did-you-mean).

    Returns the :class:`~neomd.driver.RunOutcome` of the run.
    """
    _scan_plugins()  # ADR-0002: plugins register before the Plan validates
    plan = _resolve_plan(target, overrides)
    return compile(plan, kernel=kernel, platform=platform,
                   logger=logger).run()
