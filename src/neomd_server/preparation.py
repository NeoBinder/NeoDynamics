"""Read-only accepted preparation preset provenance; no request execution yet."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json


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
    return {"schema_version": 1, "presets": [preset], "system_assembly_available": False,
            "unresolved_parameters": []}
