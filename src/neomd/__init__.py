"""
neomd — the v2 spine of NeoDynamics.

Public surface by progressive disclosure (see ``run.py``): L0
``md_run(dir)`` -> L1 scalar kwargs -> L2 full plan.  This ``__init__``
deliberately keeps no eager cross-module imports — facade symbols resolve
lazily so sub-modules develop and import independently.
"""

try:  # distribution version (versioningit derives it from git tags)
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("NeoDynamics")
except Exception:  # pragma: no cover - source tree without metadata
    __version__ = "0.2.0"

__all__ = [
    "md_run", "load_plan", "compile", "register", "validate_config",
    "check_plan_files", "plan_resume", "prepare_system", "__version__",
]


def __getattr__(name):  # lazy facade (PEP 562)
    if name == "md_run":
        from .run import md_run
        return md_run
    if name == "load_plan":
        from .plan import load_plan
        return load_plan
    if name == "compile":
        from .run import compile
        return compile
    if name == "register":
        from .registry import register
        return register
    if name == "validate_config":
        from .plan import validate_config
        return validate_config
    if name == "check_plan_files":
        from .plan import check_plan_files
        return check_plan_files
    if name == "plan_resume":
        from .resume import plan_resume
        return plan_resume
    if name == "prepare_system":
        from .prepare import prepare_system
        return prepare_system
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
