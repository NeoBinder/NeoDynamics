"""Pinned Sage 2.2.1 ligand packages from accepted chemistry, separate from bound poses."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

import rfc8785

from .component_contract import (
    ComponentContractError,
    canonical_sha256,
    generated_package_cache_key,
    validate_document,
)
from .component_gaff import _ambertools_receipt, _molecule, _ref, _uuid
from .component_package import extract_system_parameters, validate_package_assets
from .tools.port import SubprocessToolRunner

_BASE = "urn:neodyn:component-preparation:v1"
_OFFXML = "openff_unconstrained-2.2.1.offxml"
_OFFXML_SHA256 = "124f46daf213453d33d773497c35b34f29e752a42b658223355072663ba05a47"
_AMBERTOOLS_MANIFEST_SHA256 = "f571ed87f836aa2f5408b3d73bc5fcb5bd217d620dea838154f0636b3da710db"
_PROTEIN_XML_SHA256 = "062f99635106ea7e00ab98b9827299ae48ac82875569832a6dea2cb875e0b078"
_WATER_XML_SHA256 = "fc528100bfb3587f632c0492166026ed5658c897026de0408f371604a79271fc"
_ASSETS = (
    "atom_map",
    "force",
    "template",
    "offxml",
    "protein_xml",
    "water_xml",
    "conformer",
    "fractional_bond_orders",
    "provenance",
    "qc",
)
_VERSIONS = {
    "openforcefields": "2024.9.0",
    "openff-toolkit": "0.19.0",
    "openff-interchange": "0.5.4",
    "openmmforcefields": "0.16.0",
    "openmm": "8.6.0",
    "rdkit": "2026.3.1",
}
_OFFXML_DISTRIBUTION_VERSIONS = frozenset({"2024.9.0", "2026.1.0"})


class OpenffUnavailable(ComponentContractError):
    """The exact Sage/AmberTools provider is not present on this worker."""


@dataclass(frozen=True)
class OpenffPackage:
    package_bytes: bytes
    artifacts: tuple[tuple[str, bytes], ...]

    @property
    def package(self) -> dict:
        return json.loads(self.package_bytes)

    @property
    def asset_bytes(self) -> dict[str, bytes]:
        return dict(self.artifacts)


def _fail(code: str, path: str, detail: str) -> None:
    raise ComponentContractError(code, path, detail)


def _resources(runner=None):
    """Read installed bytes only after checking every version and source digest."""
    try:
        actual = {name: importlib.metadata.version(name) for name in _VERSIONS}
        if (
            actual["openforcefields"] not in _OFFXML_DISTRIBUTION_VERSIONS
            or any(
                actual[name] != version
                for name, version in _VERSIONS.items()
                if name != "openforcefields"
            )
        ):
            raise ValueError("tool versions differ")
        amber_version, amber_hashes = _ambertools_receipt(runner or SubprocessToolRunner())
        if amber_version != "26.0":
            raise ValueError("AmberTools version differs")
        manifest = [
            row["sha256"]
            for row in amber_hashes
            if row["resource_id"] == "ambertools/package-manifest"
        ]
        if manifest != [_AMBERTOOLS_MANIFEST_SHA256]:
            raise ValueError("AmberTools package inventory differs")
        offxml = (files("openforcefields") / "offxml" / _OFFXML).read_bytes()
        if hashlib.sha256(offxml).hexdigest() != _OFFXML_SHA256:
            raise ValueError("Sage bytes differ")
        protein = (
            files("openmmforcefields") / "ffxml" / "amber" / "protein.ff14SB.xml"
        ).read_bytes()
        water = (files("openmmforcefields") / "ffxml" / "amber" / "tip3p_standard.xml").read_bytes()
        if (
            hashlib.sha256(protein).hexdigest() != _PROTEIN_XML_SHA256
            or hashlib.sha256(water).hexdigest() != _WATER_XML_SHA256
        ):
            raise ValueError("shared ff14SB/TIP3P resources differ")
        from openff.toolkit.utils import AmberToolsToolkitWrapper
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        if (
            not AmberToolsToolkitWrapper.is_available()
            or "am1bcc" not in AmberToolsToolkitWrapper().supported_charge_methods
            or not hasattr(SMIRNOFFTemplateGenerator, "convert_system_to_ffxml")
        ):
            raise ValueError("required OpenFF backend is unavailable")
    except ComponentContractError as error:
        raise OpenffUnavailable(
            "capability_unavailable",
            "/method",
            f"pinned Sage provider unavailable: {type(error).__name__}",
        ) from None
    except (
        ImportError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        raise OpenffUnavailable(
            "capability_unavailable",
            "/method",
            f"pinned Sage provider unavailable: {type(error).__name__}",
        ) from None
    return offxml, protein, water, amber_hashes, actual


def openff_provider_ready() -> bool:
    """Only advertise the method when its pinned local resources are present."""
    try:
        _resources()
    except ComponentContractError:
        return False
    return True


def _system_options(seed: int, formal_charge: int, *, has_stereo: bool = False) -> dict:
    options = {
        "profile": "sage2.2.1-am1bcc-fixed-system-v1",
        "force_field": _OFFXML,
        "charge_backend": "ambertools-26.0",
        "charge_method": "am1bcc",
        "charge_normalization": "openff-normalize-to-formal-charge",
        "fractional_bond_order_backend": "ambertools-26.0",
        "fractional_bond_order_method": "am1-wiberg",
        "conformer_engine": "rdkit-ETKDGv3-MMFF94s" if has_stereo else "rdkit-ETKDGv3",
        "conformer_count": 1,
        "seed": seed,
        "net_formal_charge": formal_charge,
        "forcefield_load_order": [
            "amber/protein.ff14SB.xml",
            "amber/tip3p_standard.xml",
            "generated-sage-template",
        ],
        "system": {
            "nonbonded_method": "pme",
            "cutoff_nm": 1.0,
            "constraints": "HBonds",
            "hydrogen_mass_amu": 4.0,
            "rigid_water": True,
            "remove_cm_motion": False,
            "switch_distance_nm": None,
            "flexible_constraints": False,
            "ewald_error_tolerance": 0.0005,
            "use_dispersion_correction": True,
            "exceptions_use_periodic_boundary_conditions": False,
        },
    }
    if has_stereo:
        options["stereo_policy"] = "reviewed-CIP-preserved-before-and-after-MMFF94s"
    return options


def build_openff_package(
    state, *, package_id, artifact_ids, seed=20260926, runner=None
) -> OpenffPackage:
    """Generate one state-bound Sage/AM1-BCC package with exact retained inputs."""
    from openff.toolkit import ForceField
    from openff.toolkit.utils import AmberToolsToolkitWrapper, RDKitToolkitWrapper, ToolkitRegistry
    from openmm import Vec3, app, unit
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    validate_document("chemical-state", state)
    if state["review"]["status"] != "accepted":
        _fail("chemistry_unresolved", "/review/status", "accepted state is required")
    if type(seed) is not int or not 1 <= seed <= 0x7FFFFFFF:
        _fail("invalid_reference", "/seed", "ETKDG seed must be a positive 31-bit integer")
    _uuid(package_id, "/package_id")
    if not isinstance(artifact_ids, dict) or set(artifact_ids) != set(_ASSETS):
        _fail("invalid_reference", "/artifact_ids", "all ten asset identities are required")
    for key, identity in artifact_ids.items():
        _uuid(identity, f"/artifact_ids/{key}")
    if len(set(artifact_ids.values()) | {package_id}) != len(_ASSETS) + 1:
        _fail("invalid_reference", "/artifact_ids", "package and asset UUIDs must be distinct")

    offxml, protein_xml, water_xml, amber_hashes, tool_versions = _resources(runner)
    has_stereo = any(row["stereo"] in ("R", "S") for row in state["atoms"])
    wrapped = _molecule(state, seed, optimize_stereo=True)
    molecule = wrapped.molecule
    with tempfile.TemporaryDirectory(prefix="neomd-sage-") as directory:
        sdf_path = Path(directory) / "generation.sdf"
        wrapped.to_file(sdf_path, file_format="SDF")
    if wrapped.sdf_bytes is None or wrapped.sdf_sha256 is None:
        _fail("capability_unavailable", "/conformer", "generation conformer was not captured")
    conformer = molecule.conformers
    if conformer is None or len(conformer) != 1:
        _fail("chemistry_unresolved", "/conformer", "exactly one conformer is required")
    try:
        backend = AmberToolsToolkitWrapper()
        molecule.assign_fractional_bond_orders(
            "am1-wiberg", toolkit_registry=backend, use_conformers=conformer
        )
        molecule.assign_partial_charges(
            "am1bcc",
            toolkit_registry=backend,
            use_conformers=conformer,
            strict_n_conformers=True,
            normalize_partial_charges=True,
        )
        charges = [float(value.magnitude) for value in molecule.partial_charges]
        if (
            len(charges) != len(state["atoms"])
            or any(not math.isfinite(q) for q in charges)
            or abs(sum(charges) - state["net_formal_charge"]) > 1e-6
        ):
            raise ValueError("partial charges differ from the reviewed net charge")
        state_ids = [row["id"] for row in state["atoms"]]
        fbo = [
            {
                "atom_ids": [state_ids[bond.atom1_index], state_ids[bond.atom2_index]],
                "order": float(bond.fractional_bond_order),
            }
            for bond in molecule.bonds
        ]
        if len(fbo) != len(state["bonds"]) or any(
            not math.isfinite(row["order"]) or row["order"] <= 0 for row in fbo
        ):
            raise ValueError("fractional bond orders are incomplete")
        registry = ToolkitRegistry([RDKitToolkitWrapper(), backend])
        with tempfile.TemporaryDirectory(prefix="neomd-sage-offxml-") as directory:
            verified_offxml = Path(directory) / _OFFXML
            verified_offxml.write_bytes(offxml)
            field = ForceField(str(verified_offxml))
            direct = field.create_openmm_system(
                molecule.to_topology(),
                toolkit_registry=registry,
                charge_from_molecules=[molecule],
                partial_bond_orders_from_molecules=[molecule],
            )
            generator = SMIRNOFFTemplateGenerator(
                molecules=molecule, forcefield=str(verified_offxml)
            )
            template = generator.convert_system_to_ffxml(molecule, direct).encode()
        topology = molecule.to_topology().to_openmm()
        topology.setUnitCellDimensions(Vec3(3, 3, 3) * unit.nanometer)
        openmm_field = app.ForceField(
            *(io.BytesIO(data) for data in (protein_xml, water_xml, template))
        )
        system = openmm_field.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            hydrogenMass=4.0 * unit.amu,
            removeCMMotion=False,
            switchDistance=None,
            flexibleConstraints=False,
            ewaldErrorTolerance=0.0005,
            useDispersionCorrection=True,
            residueTemplates={},
            ignoreExternalBonds=False,
        )
        nonbonded = [
            force for force in system.getForces() if type(force).__name__ == "NonbondedForce"
        ]
        if len(nonbonded) != 1:
            raise ValueError("requires one NonbondedForce")
        nonbonded[0].setExceptionsUsePeriodicBoundaryConditions(False)
        if (
            nonbonded[0].getNonbondedMethod() != nonbonded[0].PME
            or float(nonbonded[0].getCutoffDistance().value_in_unit(unit.nanometer)) != 1.0
            or nonbonded[0].getEwaldErrorTolerance() != 0.0005
            or not nonbonded[0].getUseDispersionCorrection()
            or nonbonded[0].getUseSwitchingFunction()
            or nonbonded[0].getExceptionsUsePeriodicBoundaryConditions()
        ):
            raise ValueError("nonbonded System settings differ from accepted profile")
        force = extract_system_parameters(system, state_ids)
    except ComponentContractError:
        raise
    except Exception as error:
        raise ComponentContractError(
            "capability_unavailable",
            "/method",
            f"Sage parameterization failed: {type(error).__name__}",
        ) from None

    atom_map = {
        "format": "neomd-component-atom-map-v1",
        "atoms": [
            {
                "state_atom_id": state_id,
                "parameter_index": index,
                "parameter_atom_name": molecule.atoms[index].name,
            }
            for index, state_id in enumerate(state_ids)
        ],
    }
    options = _system_options(seed, state["net_formal_charge"], has_stereo=has_stereo)
    fbo_bytes = rfc8785.dumps({"format": "neomd-am1-wiberg-orders-v1", "bonds": fbo})
    provenance = {
        "format": "neomd-openff-provenance-v1",
        "state_sha256": canonical_sha256(state),
        "conformer_sdf_sha256": wrapped.sdf_sha256,
        "fractional_bond_orders_sha256": hashlib.sha256(fbo_bytes).hexdigest(),
        "charge_sha256": canonical_sha256(charges),
        "options": options,
    }
    qc = {
        "format": "neomd-component-qc-v1",
        "status": "integrity_passed",
        "checks": [
            "reviewed graph and atom order retained",
            "one seeded generation conformer reused for AM1-Wiberg and AM1-BCC",
            "explicit AmberTools backend assigned integral-net-charge partial charges",
            "pinned Sage 2.2.1 direct System converted to a retained template",
            "ff14SB, TIP3P and template built the fixed OpenMM System profile",
        ],
    }
    payloads = {
        "atom_map": rfc8785.dumps(atom_map),
        "force": rfc8785.dumps(force),
        "template": template,
        "offxml": offxml,
        "protein_xml": protein_xml,
        "water_xml": water_xml,
        "conformer": wrapped.sdf_bytes,
        "fractional_bond_orders": fbo_bytes,
        "provenance": rfc8785.dumps(provenance),
        "qc": rfc8785.dumps(qc),
    }
    refs = {
        key: _ref(
            artifact_ids[key],
            "atom_map" if key == "atom_map" else "qc_report" if key == "qc" else "parameter_asset",
            payloads[key],
            "application/json"
            if key in ("atom_map", "force", "fractional_bond_orders", "provenance", "qc")
            else "application/octet-stream",
        )
        for key in _ASSETS
    }
    resources = amber_hashes + [
        {"resource_id": "openff/openff_unconstrained-2.2.1.offxml", "sha256": _OFFXML_SHA256},
        {
            "resource_id": "amber/protein.ff14SB.xml",
            "sha256": hashlib.sha256(protein_xml).hexdigest(),
        },
        {
            "resource_id": "amber/tip3p_standard.xml",
            "sha256": hashlib.sha256(water_xml).hexdigest(),
        },
    ]
    charge = sum((Decimal(row["charge_e"]) for row in force["atoms"]), Decimal(0))
    package = {
        "schema_id": f"{_BASE}:parameter-package",
        "schema_version": 1,
        "package_id": package_id,
        "state_id": state["state_id"],
        "state_sha256": canonical_sha256(state),
        "origin": "generated",
        "method": {
            "method_id": "openff.sage2.2.1.am1bcc",
            "version": "openff-2.2.1",
            "charge_method_id": "am1bcc",
            "tool_versions": [
                {"tool": "ambertools", "version": "26.0"},
                *({"tool": name, "version": version} for name, version in tool_versions.items()),
            ],
            "resource_hashes": resources,
            "options_sha256": canonical_sha256(options),
            "conformer_sha256": wrapped.sdf_sha256,
            "seed": seed,
        },
        "atom_map": refs["atom_map"],
        "assets": [refs[key] for key in _ASSETS if key not in ("atom_map", "qc")],
        "compatibility": {
            "protein_family": "amber-ff14sb",
            "ligand_family": "openff-sage-2.2.1",
            "water_family": "tip3p",
            "ion_family": "amber-tip3p",
            "combining_rule": "lorentz-berthelot",
            "electrostatic_14_scale": 0.8333333333,
            "vdw_14_scale": 0.5,
            "unit_system": "openmm_nm_kj_mol_e",
            "nonbonded_method": "pme",
            "cutoff_nm": 1.0,
            "switch_distance_nm": None,
            "constraints": "HBonds",
            "hydrogen_mass_amu": 4.0,
            "virtual_sites": False,
        },
        "charge_sum_e": float(charge),
        "qc": {"status": "integrity_passed", "report": refs["qc"]},
    }
    assets = {artifact_ids[key]: payloads[key] for key in _ASSETS}
    validate_package_assets(state, package, assets)
    encoded = rfc8785.dumps(package)
    if len(encoded) > 1048576:
        _fail("oversized_payload", "", "package control JSON exceeds 1 MiB")
    return OpenffPackage(encoded, tuple(sorted(assets.items())))


def openff_cache_key(package):
    """Keep Sage/charge/FBO/conformer identity separate from GAFF and bound poses."""
    validate_document("parameter-package", package)
    if (
        package["origin"] != "generated"
        or package["method"]["method_id"] != "openff.sage2.2.1.am1bcc"
    ):
        _fail("invalid_reference", "/method", "requires a generated Sage package")
    return generated_package_cache_key(package["state_sha256"], package["method"])
