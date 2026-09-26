"""Read-only accepted preparation preset provenance; no request execution yet."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
from copy import deepcopy


def preparation_catalog():
    """Describe exact installed resources and accepted user-editable defaults.

    Resource names are opaque catalog identifiers, not client filesystem paths.
    Missing dependencies fail explicitly; this catalog does not certify a System.
    """
    package = "openmmforcefields"
    version = importlib.metadata.version(package)
    resources = []
    for role, name in (("protein", "protein.ff14SB.xml"), ("water_and_ions", "tip3p_standard.xml")):
        data = (importlib.resources.files(package) / "ffxml" / "amber" / name).read_bytes()
        resources.append({"role": role, "resource_id": f"amber/{name}",
                          "package": package, "package_version": version,
                          "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})
    preset = {"preset_id": "protein.ff14sb.tip3p", "resources": resources,
              "accepted_parameters": {"water_geometry": "tip3p", "positive_ion": "Na+",
                                      "negative_ion": "Cl-", "ionic_strength_mol_per_l": 0.15,
                                      "neutralize": True, "box_shape": "cubic",
                                      "padding_nm": 1.0, "ph": 7.0,
                                      "existing_water": "retain_and_solvate",
                                      "hydrogen_mass_amu": 4.0, "constraints": "HBonds",
                                      "rigid_water": True, "remove_cm_motion": False,
                                      "nonbonded_method": "pme", "nonbonded_cutoff_nm": 1.0,
                                      "ewald_error_tolerance": 0.0005,
                                      "use_dispersion_correction": True,
                                      "switch_distance_nm": None,
                                      "exceptions_use_periodic_boundary_conditions": False,
                                      "flexible_constraints": False},
              "editable_parameters": ["ionic_strength_mol_per_l", "padding_nm", "ph"],
              "repair": {"hydrogens": True, "standard_residue_heavy_atoms": True,
                         "missing_stretches": "block", "nonstandard_residues": "block",
                         "ligands_metals_nucleic_acids": "block"}}
    encoded = json.dumps(preset, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    preset["preset_revision_sha256"] = hashlib.sha256(encoded).hexdigest()
    limits = {"max_source_bytes": 67108864, "max_result_bytes": 536870912,
              "time_limit_seconds": 3600, "memory_limit_bytes": 4294967296,
              "max_concurrent_operations": 1, "max_queued_per_owner": 8}
    # Clients pin this exact identity in source descriptors; the version must
    # be discovered here rather than hardcoded at capture time.
    import openmm
    loader = {"name": "openmm.PDBxFile", "version": openmm.__version__,
              "atom_order": "preparation_input"}
    # Qualified by tests/server/test_preparation_operations.py: real repair, H
    # completion, retained input water, solvent/ions, System assembly and
    # independent verification all pass through the isolated executor.
    return {"schema_version": 1, "presets": [preset], "system_assembly_available": True,
            "loader": loader, "limits": limits, "unresolved_parameters": []}


def topology_preparation_catalog():
    """The explicit-bond ff14SB/TIP3P/GAFF profile qualified on 3HTB."""
    original = preparation_catalog()
    preset = deepcopy(original["presets"][0])
    preset["preset_id"] = "protein.ff14sb.tip3p.gaff.explicit"
    preset["repair"] = {
        "hydrogens": True,
        "standard_residue_heavy_atoms": False,
        "missing_stretches": "block",
        "nonstandard_residues": "declared_ligand_graph_only",
        "ligands_metals_nucleic_acids": "declared_ligands_only",
    }
    preset["source_topology"] = {
        "schema_id": "urn:neodyn:molecular-topology:v1",
        "coverage": "complete_asserted",
        "ligand_role": "residue.properties.parameter_role=ligand",
        "box_policy": "orthorhombic_source_box_meets_padding",
        "covalent_protein_ligand": "unsupported",
    }
    preset["ligand_parameterizer"] = {
        "name": "GAFFTemplateGenerator",
        "openmmforcefields_version": importlib.metadata.version("openmmforcefields"),
        "openff_toolkit_version": importlib.metadata.version("openff-toolkit"),
    }
    preset["accepted_parameters"]["ionic_strength_mol_per_l"] = 0.1
    preset["accepted_parameters"]["padding_nm"] = 1.0
    preset["accepted_parameters"]["box_shape"] = "source_orthorhombic"
    preset.pop("preset_revision_sha256")
    encoded = json.dumps(preset, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    preset["preset_revision_sha256"] = hashlib.sha256(encoded).hexdigest()
    return {
        "schema_version": 2,
        "presets": [preset],
        "system_assembly_available": True,
        "loader": {"name": "neomd.canonical_topology_adapter", "version": "1",
                   "atom_order": "preparation_input"},
        "limits": original["limits"],
        "unresolved_parameters": [],
    }
