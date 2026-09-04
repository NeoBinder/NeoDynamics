"""
neomd.tools — the external-tool seam.

* :mod:`neomd.tools.port` — ``ToolRunner`` (isolated execution +
  diagnostics), ``ToolResult`` / ``ToolError``, the in-process
  ``FakeToolRunner``, and the minimal ``ChargeBackend`` / ``ParamBackend``
  protocols.
* :mod:`neomd.tools.antechamber` — GAFF/antechamber knowledge:
  ``AntechamberBackend``, the thin ``GAFFTemplateGenerator``,
  ``rename_atoms_by_template``, and ``build`` (the ``ForceFieldBuilder``
  seam entry).
"""

from neomd.tools.antechamber import (
    AntechamberBackend,
    GAFFTemplateGenerator,
    build,
    register_gaff_generator,
    rename_atoms_by_template,
    sys_params_from_config,
)
from neomd.tools.port import (
    ChargeBackend,
    FakeCall,
    FakeToolRunner,
    ParamBackend,
    SubprocessToolRunner,
    ToolError,
    ToolResult,
    ToolRunner,
)

__all__ = [
    "AntechamberBackend",
    "ChargeBackend",
    "FakeCall",
    "FakeToolRunner",
    "GAFFTemplateGenerator",
    "ParamBackend",
    "SubprocessToolRunner",
    "ToolError",
    "ToolResult",
    "ToolRunner",
    "build",
    "register_gaff_generator",
    "rename_atoms_by_template",
    "sys_params_from_config",
]
