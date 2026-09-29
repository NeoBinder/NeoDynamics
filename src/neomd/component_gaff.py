"""State-bound GAFF2/ABCG2 ligand packages from a separate generation conformer.

Only the reviewed ChemicalState graph enters this provider. A bound pose is
neither read nor used for charge generation; assembly consumes the retained
force asset and never needs antechamber.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import UUID
from xml.etree import ElementTree

import rfc8785

from .component_contract import (
    ComponentContractError,
    canonical_sha256,
    generated_package_cache_key,
    validate_document,
)
from .component_package import extract_system_parameters, validate_package_assets
from .tools.antechamber import AntechamberBackend
from .tools.port import SubprocessToolRunner, ToolError, ToolRunner

_BASE = "urn:neodyn:component-preparation:v1"
_GAFF = "gaff-2.2.20"
_ASSETS = ("atom_map", "force", "template", "gaff_xml", "gaff_dat",
           "protein_xml", "water_xml", "conformer", "provenance", "qc")
_TIMEOUT_SECONDS = 180
_STEREO_TIMEOUT_SECONDS = 600


class GaffUnavailable(ComponentContractError):
    """The pinned GAFF provider cannot run on this host."""


@dataclass(frozen=True)
class GaffPackage:
    package_bytes: bytes
    artifacts: tuple[tuple[str, bytes], ...]

    @property
    def package(self) -> dict:
        return json.loads(self.package_bytes)

    @property
    def asset_bytes(self) -> dict[str, bytes]:
        return dict(self.artifacts)


class _BoundedRunner(SubprocessToolRunner):
    """Preserve ToolRunner's per-call directory isolation with a time limit."""

    def __init__(self, timeout_seconds=_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def _spawn(self, command, cwd, env):
        process = subprocess.Popen(command, cwd=str(cwd), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=os.name == "posix")
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            # AmberTools wrappers may leave sqm running after their own shell
            # is killed; terminate the whole isolated subprocess group.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            raise ToolError("external parameterization timed out", command=command) from error
        return process.returncode, stdout or "", stderr or ""


class _TracingRunner(ToolRunner):
    def __init__(self, inner):
        self.inner = inner
        self.commands = []

    def which(self, name):
        return self.inner.which(name)

    def _spawn(self, command, cwd, env):
        raise AssertionError("tracing wrapper delegates run directly")

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        return self.inner.run(command, **kwargs)


class _PinnedMolecule:
    """Prevent the legacy backend from replacing the prepared conformer."""

    def __init__(self, molecule):
        self.molecule = molecule
        self.sdf_sha256 = None
        self.sdf_bytes = None

    def __getattr__(self, key):
        return getattr(self.molecule, key)

    def generate_conformers(self, **kwargs):
        if len(self.molecule.conformers or []) != 1:
            raise ValueError("exactly one parameter-generation conformer is required")

    def to_file(self, filename, **kwargs):
        self.molecule.to_file(filename, **kwargs)
        self.sdf_bytes = Path(filename).read_bytes()
        self.sdf_sha256 = hashlib.sha256(self.sdf_bytes).hexdigest()


def _fail(code, path, detail):
    raise ComponentContractError(code, path, detail)


def _uuid(value, path):
    try:
        if not isinstance(value, str) or UUID(value).int == 0 or str(UUID(value)) != value:
            raise ValueError("noncanonical")
    except (ValueError, AttributeError, TypeError):
        _fail("invalid_reference", path, "requires a canonical nonnil UUID")


def _ref(identity, kind, data, media_type):
    return {"schema_id": f"{_BASE}:artifact-reference", "schema_version": 1,
            "artifact_id": identity, "kind": kind,
            "content_sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "decoded_size_bytes": len(data),
            "encoding": "identity", "media_type": media_type}


def _ambertools_receipt(runner):
    """Verify both wrappers and real executables against their installed manifest."""
    paths = {name: runner.which(name) for name in ("antechamber", "parmchk2")}
    if any(path is None or not Path(path).is_file() for path in paths.values()):
        raise GaffUnavailable("capability_unavailable", "/method",
                              "antechamber and parmchk2 executables are required")
    roots = {Path(path).resolve().parent.parent for path in paths.values()}
    if len(roots) != 1:
        raise GaffUnavailable("capability_unavailable", "/method",
                              "AmberTools binaries do not share one verified installation")
    root = roots.pop()
    manifests = list((root / "conda-meta").glob("ambertools-*.json"))
    if len(manifests) != 1:
        raise GaffUnavailable("capability_unavailable", "/method",
                              "AmberTools package receipt is unavailable")
    try:
        manifest = json.loads(manifests[0].read_text())
        inventory = {row["_path"]: row["sha256"]
                     for row in manifest["paths_data"]["paths"] if "sha256" in row}
        if manifest["name"] != "ambertools" or manifest["version"] != "26.0":
            raise ValueError("different AmberTools version")
        # A traced JZ4 ABCG2 run executes these helpers in addition to the
        # top-level pair. Verify both wrappers and wrapped programs.
        requested = ["amber.sh", "bin/sqm"]
        for name in ("antechamber", "parmchk2", "am1bcc", "atomtype", "bondtype"):
            requested.extend((f"bin/{name}", f"bin/wrapped_progs/{name}"))
        resources = []
        for relative in requested:
            data = (root / relative).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != inventory[relative]:
                raise ValueError("AmberTools installed bytes differ")
            resources.append({"resource_id": f"ambertools/{relative}", "sha256": digest})
        manifest_digest = hashlib.sha256(rfc8785.dumps({
            "name": manifest["name"], "version": manifest["version"],
            "build": manifest["build"], "paths": sorted(inventory.items()),
        })).hexdigest()
        resources.append({"resource_id": "ambertools/package-manifest",
                          "sha256": manifest_digest})
        if any(Path(paths[name]).resolve() != (root / "bin" / name).resolve() for name in paths):
            raise ValueError("executed wrappers differ from manifest")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise GaffUnavailable("capability_unavailable", "/method",
                              f"AmberTools installation is unverifiable: {type(error).__name__}") from None
    return manifest["version"], resources


def _scales(xml_bytes):
    try:
        nonbonded = ElementTree.fromstring(xml_bytes).find("NonbondedForce")
        return (float(nonbonded.attrib["coulomb14scale"]),
                float(nonbonded.attrib["lj14scale"]))
    except (ElementTree.ParseError, AttributeError, KeyError, TypeError, ValueError):
        raise GaffUnavailable("capability_unavailable", "/compatibility",
                              "force-field 1-4 scales are unavailable") from None


def _molecule(state, seed, *, optimize_stereo=False):
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Chem import AllChem

    graph = Chem.RWMol()
    indexes = {}
    for row in state["atoms"]:
        atom = Chem.Atom(row["element"])
        atom.SetFormalCharge(row["formal_charge"])
        atom.SetNoImplicit(True)
        indexes[row["id"]] = graph.AddAtom(atom)
    orders = {"single": Chem.BondType.SINGLE, "double": Chem.BondType.DOUBLE,
              "triple": Chem.BondType.TRIPLE, "aromatic": Chem.BondType.AROMATIC}
    for row in state["bonds"]:
        left, right = row["atom_ids"]
        graph.AddBond(indexes[left], indexes[right], orders[row["order"]])
    rdkit_molecule = graph.GetMol()
    try:
        Chem.SanitizeMol(rdkit_molecule)
        potential = Chem.FindPotentialStereo(rdkit_molecule)
        atom_stereo = {index for index, row in enumerate(state["atoms"])
                       if row["stereo"] in ("R", "S")}
        possible_atoms = {item.centeredOn for item in potential
                          if str(item.type) == "Atom_Tetrahedral"}
        if atom_stereo != possible_atoms or any(
            row["stereo"] != "none" for row in state["bonds"]
        ) or any(str(item.type) == "Bond_Double" for item in potential):
            raise ValueError("reviewed stereochemistry is incomplete or unsupported")
        for index in atom_stereo:
            atom = rdkit_molecule.GetAtomWithIdx(index)
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            Chem.AssignStereochemistry(rdkit_molecule, cleanIt=True, force=True)
            if atom.GetProp("_CIPCode") != state["atoms"][index]["stereo"]:
                atom.InvertChirality()
                Chem.AssignStereochemistry(rdkit_molecule, cleanIt=True, force=True)
            if atom.GetProp("_CIPCode") != state["atoms"][index]["stereo"]:
                raise ValueError("reviewed CIP label cannot be assigned")
        settings = AllChem.ETKDGv3()
        settings.randomSeed = seed
        if AllChem.EmbedMolecule(rdkit_molecule, settings) != 0:
            raise ValueError("embedding failed")
        if optimize_stereo and atom_stereo:
            if AllChem.MMFFGetMoleculeProperties(rdkit_molecule, mmffVariant="MMFF94s") is None:
                raise ValueError("MMFF94s parameters are unavailable")
            if AllChem.MMFFOptimizeMolecule(
                rdkit_molecule, mmffVariant="MMFF94s", maxIters=2000
            ) != 0:
                raise ValueError("MMFF94s conformer relaxation did not converge")
        observed = Chem.Mol(rdkit_molecule)
        Chem.AssignAtomChiralTagsFromStructure(observed, replaceExistingTags=True)
        Chem.AssignStereochemistryFrom3D(observed)
        Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
        for index in atom_stereo:
            atom = observed.GetAtomWithIdx(index)
            if not atom.HasProp("_CIPCode") or atom.GetProp("_CIPCode") != state["atoms"][index]["stereo"]:
                raise ValueError("generated conformer changed reviewed stereochemistry")
        molecule = Molecule.from_rdkit(rdkit_molecule, hydrogens_are_explicit=True)
    except Exception as error:
        raise ComponentContractError("chemistry_unresolved", "/atoms",
                                     f"generation conformer failed: {type(error).__name__}") from None
    if len(molecule.atoms) != len(state["atoms"]) or len(molecule.conformers or []) != 1:
        _fail("mapping_mismatch", "/atoms", "OpenFF changed reviewed atom inventory")
    for atom, row in zip(molecule.atoms, state["atoms"]):
        if atom.symbol != row["element"] or int(atom.formal_charge.magnitude) != row["formal_charge"]:
            _fail("mapping_mismatch", "/atoms", "OpenFF changed reviewed atom order or charge")
        if atom.stereochemistry != (None if row["stereo"] == "none" else row["stereo"]):
            _fail("mapping_mismatch", "/atoms", "OpenFF changed reviewed stereochemistry")
        atom.name = f"{row['element']}{row['id']}"
    return _PinnedMolecule(molecule)


def build_gaff_package(state, *, package_id, artifact_ids, seed=20260926,
                       runner: ToolRunner | None = None) -> GaffPackage:
    """Generate exact pinned GAFF2/ABCG2 terms for one accepted state.

    ``artifact_ids`` names all ten keys in ``_ASSETS``. Caller assigns opaque
    UUIDs and persists returned bytes.
    A supplied runner is intended for controlled tests and owns its own bound.
    """
    from openmm import Vec3, app, unit
    from openmmforcefields.generators import GAFFTemplateGenerator as LibraryGenerator

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

    has_stereo = any(row["stereo"] in ("R", "S") for row in state["atoms"])
    active = _TracingRunner(runner or _BoundedRunner(
        _STEREO_TIMEOUT_SECONDS if has_stereo else _TIMEOUT_SECONDS))
    amber_version, binary_hashes = _ambertools_receipt(active)
    if _GAFF not in LibraryGenerator.INSTALLED_FORCEFIELDS:
        raise GaffUnavailable("capability_unavailable", "/method",
                              "pinned GAFF 2.2.20 resource is unavailable")
    backend = AntechamberBackend(active, gaff_version=_GAFF)
    gaff_dat = Path(backend.gaff_dat_filename).read_bytes()
    gaff_xml_path = Path(backend.gaff_dat_filename).parent.parent / "ffxml" / f"{_GAFF}.xml"
    gaff_xml = gaff_xml_path.read_bytes()
    from importlib.resources import files
    protein_xml = (files("openmmforcefields") / "ffxml" / "amber" / "protein.ff14SB.xml").read_bytes()
    water_xml = (files("openmmforcefields") / "ffxml" / "amber" / "tip3p_standard.xml").read_bytes()
    scales = _scales(gaff_xml)
    if scales != _scales(protein_xml) or scales != _scales(water_xml):
        _fail("incompatible_package", "/compatibility", "GAFF, ff14SB and TIP3P 1-4 scales differ")
    molecule = _molecule(state, seed, optimize_stereo=True)
    try:
        template = backend.ffxml(molecule, residue_name="LIG").encode()
        if molecule.sdf_sha256 is None or molecule.sdf_bytes is None:
            _fail("invalid_reference", "/method", "generation SDF was not captured")
        topology = molecule.molecule.to_topology().to_openmm()
        topology.setUnitCellDimensions(Vec3(3, 3, 3) * unit.nanometer)
        # Existing neomd/tools/antechamber.build starts with protein and water
        # before lazily adding GAFF and the generated frcmod template. OpenMM
        # parameter precedence depends on this order (JZ4 c3-ca is one case).
        field = app.ForceField(*(io.BytesIO(data) for data in
                                 (protein_xml, water_xml, gaff_xml, template)))
        system = field.createSystem(
            topology, nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds, rigidWater=True,
            hydrogenMass=4.0 * unit.amu, removeCMMotion=False,
            switchDistance=None, flexibleConstraints=False,
            ewaldErrorTolerance=0.0005, useDispersionCorrection=True,
            residueTemplates={}, ignoreExternalBonds=False)
        nonbonded = [force for force in system.getForces()
                     if type(force).__name__ == "NonbondedForce"]
        if len(nonbonded) != 1:
            raise ValueError("requires one NonbondedForce")
        nonbonded[0].setExceptionsUsePeriodicBoundaryConditions(False)
        if (nonbonded[0].getNonbondedMethod() != nonbonded[0].PME
                or float(nonbonded[0].getCutoffDistance().value_in_unit(unit.nanometer)) != 1.0
                or nonbonded[0].getEwaldErrorTolerance() != 0.0005
                or not nonbonded[0].getUseDispersionCorrection()
                or nonbonded[0].getUseSwitchingFunction()
                or nonbonded[0].getExceptionsUsePeriodicBoundaryConditions()):
            raise ValueError("nonbonded System settings differ from accepted profile")
    except ComponentContractError:
        raise
    except Exception as error:
        raise ComponentContractError("capability_unavailable", "/method",
                                     f"GAFF parameterization failed: {type(error).__name__}") from None
    atom_ids = [row["id"] for row in state["atoms"]]
    force = extract_system_parameters(system, atom_ids)
    force_bytes = rfc8785.dumps(force)
    atom_map = {"format": "neomd-component-atom-map-v1", "atoms": [
        {"state_atom_id": state_id, "parameter_index": index,
         "parameter_atom_name": molecule.molecule.atoms[index].name}
        for index, state_id in enumerate(atom_ids)]}
    options = {"profile": "gaff2-abcg2-fixed-system-v1", "gaff": _GAFF,
               "atom_type": "gaff2", "charge_type": "abcg2", "seed": seed,
               "charge_correction": "net_formal_charge_minus_raw_sum_times_abs_charge_over_sum_abs_charge_if_nonzero",
               "net_formal_charge": state["net_formal_charge"],
               "atom_formal_charges": [row["formal_charge"] for row in state["atoms"]],
               "conformer_engine": "rdkit-ETKDGv3-MMFF94s" if has_stereo else "rdkit-ETKDGv3",
               "residue_name": "LIG",
               "forcefield_load_order": ["amber/protein.ff14SB.xml",
                                         "amber/tip3p_standard.xml",
                                         f"gaff/{_GAFF}.xml", "generated-frcmod-template"],
               "system": {"nonbonded_method": "pme", "cutoff_nm": 1.0,
                          "constraints": "HBonds", "hydrogen_mass_amu": 4.0,
                          "rigid_water": True, "remove_cm_motion": False,
                          "switch_distance_nm": None, "flexible_constraints": False,
                          "ewald_error_tolerance": 0.0005,
                          "use_dispersion_correction": True,
                          "exceptions_use_periodic_boundary_conditions": False},
               "commands": active.commands}
    if has_stereo:
        options["stereo_policy"] = "reviewed-CIP-preserved-before-and-after-MMFF94s"
    provenance = {"format": "neomd-gaff-provenance-v1",
                  "state_sha256": canonical_sha256(state),
                  "conformer_sdf_sha256": molecule.sdf_sha256,
                  "options": options}
    qc = {"format": "neomd-component-qc-v1", "status": "integrity_passed",
          "checks": ["reviewed graph and atom order retained",
                     "seeded generation conformer captured independently of pose",
                     "ABCG2 charges corrected proportionally to absolute charge for exact net charge",
                     "ff14SB, TIP3P, GAFF2 and template load order retained",
                     "GAFF2/ABCG2 template and complete OpenMM terms retained"]}
    payloads = {"atom_map": rfc8785.dumps(atom_map), "force": force_bytes,
                "template": template, "gaff_xml": gaff_xml, "gaff_dat": gaff_dat,
                "protein_xml": protein_xml, "water_xml": water_xml,
                "conformer": molecule.sdf_bytes,
                "provenance": rfc8785.dumps(provenance), "qc": rfc8785.dumps(qc)}
    refs = {key: _ref(artifact_ids[key],
                      "atom_map" if key == "atom_map" else "qc_report" if key == "qc" else "parameter_asset",
                      payloads[key], "application/json" if key in ("atom_map", "force", "provenance", "qc")
                      else "application/octet-stream") for key in _ASSETS}
    charge = sum((Decimal(row["charge_e"]) for row in force["atoms"]), Decimal(0))
    package = {"schema_id": f"{_BASE}:parameter-package", "schema_version": 1,
               "package_id": package_id, "state_id": state["state_id"],
               "state_sha256": canonical_sha256(state), "origin": "generated",
               "method": {"method_id": "amber.gaff2.abcg2", "version": _GAFF,
                          "charge_method_id": "abcg2",
                          "tool_versions": [
                              {"tool": "ambertools", "version": amber_version},
                              {"tool": "openmm", "version": importlib.metadata.version("openmm")},
                              {"tool": "openmmforcefields", "version": importlib.metadata.version("openmmforcefields")},
                              {"tool": "openff-toolkit", "version": importlib.metadata.version("openff-toolkit")},
                              {"tool": "rdkit", "version": importlib.metadata.version("rdkit")}],
                          "resource_hashes": binary_hashes + [
                              {"resource_id": f"gaff/{_GAFF}.xml", "sha256": hashlib.sha256(gaff_xml).hexdigest()},
                              {"resource_id": f"gaff/{_GAFF}.dat", "sha256": hashlib.sha256(gaff_dat).hexdigest()},
                              {"resource_id": "amber/protein.ff14SB.xml", "sha256": hashlib.sha256(protein_xml).hexdigest()},
                              {"resource_id": "amber/tip3p_standard.xml", "sha256": hashlib.sha256(water_xml).hexdigest()}],
                          "options_sha256": canonical_sha256(options),
                          "conformer_sha256": molecule.sdf_sha256, "seed": seed},
               "atom_map": refs["atom_map"],
               "assets": [refs[key] for key in ("force", "template", "gaff_xml", "gaff_dat",
                                                  "protein_xml", "water_xml", "conformer", "provenance")],
               "compatibility": {"protein_family": "amber-ff14sb", "ligand_family": "gaff2",
                                 "water_family": "tip3p", "ion_family": "amber-tip3p",
                                 "combining_rule": "lorentz-berthelot",
                                 "electrostatic_14_scale": scales[0],
                                 "vdw_14_scale": scales[1],
                                 "unit_system": "openmm_nm_kj_mol_e",
                                 "nonbonded_method": "pme", "cutoff_nm": 1.0,
                                 "switch_distance_nm": None, "constraints": "HBonds",
                                 "hydrogen_mass_amu": 4.0, "virtual_sites": False},
               "charge_sum_e": float(charge),
               "qc": {"status": "integrity_passed", "report": refs["qc"]}}
    assets = {artifact_ids[key]: payloads[key] for key in _ASSETS}
    validate_package_assets(state, package, assets)
    encoded = rfc8785.dumps(package)
    if len(encoded) > 1048576:
        _fail("oversized_payload", "", "package control JSON exceeds 1 MiB")
    return GaffPackage(encoded, tuple(sorted(assets.items())))


def gaff_cache_key(package):
    """Frozen v1 generated-cache key; output assets and bound poses are excluded."""
    validate_document("parameter-package", package)
    if package["origin"] != "generated" or package["method"]["method_id"] != "amber.gaff2.abcg2":
        _fail("invalid_reference", "/method", "requires a generated GAFF package")
    return generated_package_cache_key(package["state_sha256"], package["method"])
