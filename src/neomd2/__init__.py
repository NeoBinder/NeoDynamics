"""neomd2 — the v2 spine of NeoDynamics.

Public surface (progressive disclosure, see run.py):

    L0  md_run(dir)                     zero-config start
    L1  md_run(dir, steps=..., ...)     kwargs deepen the plan
    L2  Plan.from_dict(full_spec)       everything is a plan

This package __init__ deliberately keeps NO eager cross-module imports so the
sub-modules (plan, kernel, restraints, ...) can be developed and imported
independently; facade symbols resolve lazily.
"""

__version__ = "0.2.0.dev0"

__all__ = ["md_run", "load_plan", "compile", "register", "__version__"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
