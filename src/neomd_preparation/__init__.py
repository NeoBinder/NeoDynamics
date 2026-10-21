"""Scientific system preparation for the NeoDynamics SDK.

This in-repo sibling package turns reviewed chemistry (sources, topology
records, component states, parameter packages) into immutable prepared bundles
and OpenMM Systems. It depends on `neomd` one way only; the package
shares the engine distribution version. Contract schemas remain engine-owned and are read through
`neomd.contracts`.
"""

from __future__ import annotations

from neomd import __version__  # noqa: F401  (one distribution, one version)

# Public facade. Modules stay importable directly; package-level names are the
# documented entry points used by examples and integrators.
_PUBLIC = {
    "SystemBundle": ("neomd.system", "SystemBundle"),
    "prepare_system": ("neomd_preparation.workflow", "prepare_system"),
    "verify_prepared_system": ("neomd_preparation.workflow", "verify_prepared_system"),
    "export_system_bundle": ("neomd_preparation.workflow", "export_system_bundle"),
    "load_complex": ("neomd_preparation.workflow", "load_complex"),
    "ForceFieldBuilder": ("neomd_preparation.workflow", "ForceFieldBuilder"),
    "PlainForceFieldBuilder": ("neomd_preparation.workflow", "PlainForceFieldBuilder"),
    "custom_bonds": ("neomd_preparation.workflow", "custom_bonds"),
    "custom_addH": ("neomd_preparation.workflow", "custom_addH"),
    "system_from_amber": ("neomd_preparation.workflow", "system_from_amber"),
    "system_from_gromacs": ("neomd_preparation.workflow", "system_from_gromacs"),
    "sys_params_from_config": ("neomd_preparation.workflow", "sys_params_from_config"),
    "ProteinSource": ("neomd_preparation.source", "ProteinSource"),
    "ProteinRepair": ("neomd_preparation.source", "ProteinRepair"),
    "validate_protein_source": ("neomd_preparation.source", "validate_protein_source"),
    "repair_standard_residues": ("neomd_preparation.source", "repair_standard_residues"),
    "prepare_protein_system": ("neomd_preparation.protein.pipeline", "prepare_protein_system"),
    "verify_protein_preparation": ("neomd_preparation.verify.pipeline", "verify_protein_preparation"),
    "prepare_explicit_topology": ("neomd_preparation.topology.explicit", "prepare_explicit_topology"),
    "verify_explicit_topology_preparation": (
        "neomd_preparation.topology.explicit",
        "verify_explicit_topology_preparation",
    ),
    "read_prepared_bundle": ("neomd_preparation.bundle", "read_prepared_bundle"),
    "PreparationLineage": ("neomd_preparation.bundle", "PreparationLineage"),
    "BoundedRunner": ("neomd_preparation.components.providers.gaff", "BoundedRunner"),
    "GAFF_VERSION": ("neomd_preparation.components.providers.gaff", "GAFF_VERSION"),
    "ambertools_receipt": ("neomd_preparation.components.providers.gaff", "ambertools_receipt"),
    "topology_from_document": ("neomd_preparation.components.qc", "topology_from_document"),
}

__all__ = sorted([*_PUBLIC, "errors", "integration"])


def __getattr__(name):
    try:
        module, attribute = _PUBLIC[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    from importlib import import_module

    return getattr(import_module(module), attribute)


def __dir__():
    return sorted(set(globals()) | set(_PUBLIC))
