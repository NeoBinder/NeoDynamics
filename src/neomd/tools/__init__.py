"""neomd.tools — the external-tool seam (v2 migration plan §5 item 2.4).

* :mod:`neomd.tools.port` — ``ToolRunner`` (isolated execution +
  diagnostics), ``ToolResult`` / ``ToolError``, the in-process
  ``FakeToolRunner``, and the minimal ``ChargeBackend`` / ``ParamBackend``
  protocols.
* :mod:`neomd.tools.antechamber` — the GAFF/antechamber knowledge ported
  verbatim from v1: ``AntechamberBackend``, the thin openmmforcefields
  ``GAFFTemplateGenerator`` subclass, ``rename_atoms_by_template`` (the
  rename-after-match difference point that replaces v1's vendored openmm
  copy) and ``build`` (the ``ForceFieldBuilder`` seam entry playing v1's
  ``ComplexForceField`` role).

Later: ``neomd.tools.orca`` (resp2_orca workflow, plan item 2.5).
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
