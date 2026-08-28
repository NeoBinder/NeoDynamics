"""system — kernel-agnostic system description + the preparation workflow
(v2 migration plan §5 Phase 2 item 2.3).

Two halves with two different import disciplines:

* :class:`SystemBundle` — pure data + loading + validation.  It NEVER imports
  openmm: the openmm kernel deserializes ``system.xml`` itself (see
  ``kernel/openmm.py``), so the bundle only carries file paths, ligand
  molecules (openff, lazily) and the *modification IR* (raw barostat dict +
  normalized particle-mass overrides).  ``run.compile`` is the consumer that
  turns the same IR into a :class:`~neomd2.kernel.port.KernelSpec`; the
  bundle exists so callers can inspect/describe a system without a kernel.
* :func:`prepare_system` — the port of ``bin/prepare_openmm_system.py``
  (``prepare_system``/``make_system``).  This is a WORKFLOW, not a core-spine
  module, so it imports openmm directly at call time (the plan's "only
  kernel/openmm.py imports openmm in core" rule refers to the spine; system
  preparation, like the openmm adapter, lives at the openmm boundary).  The
  openmm import is nonetheless lazy (via :func:`_openmm`) so that importing
  :class:`SystemBundle` alone never drags the engine in.

The fgroup write-back is DEAD (plan §2)
---------------------------------------
v1's ``NeoSystem.system_add_restraints`` mutated the freshly deserialized
openmm System (adding restraint forces, assigning force groups) and then
wrote the assigned groups back into the user's config
(``config.restraint[name]["fgroup"] = fgroup`` — neosystem.py:122).  v2 kills
both halves:

* restraints never touch a SystemBundle — nothing in this module adds a
  Force, assigns a force group, or knows the word "fgroup";
* plan restraint entries flow through the registry knowledge triples to
  ``kernel.install_bias`` (``driver.drive``), and the assigned force-group
  ids come back as ``RunOutcome.fgroups`` (name -> list[int]) — a return
  value of the interface, never a config mutation.

The tools seam (parallel workstream, plan §5 items 2.4/2.5)
-----------------------------------------------------------
The heavy parameterization knowledge (ComplexForceField, GAFF template
generation, rename-after-match template matching) lives in
``neomd2.tools.antechamber``.  :func:`prepare_system` takes that layer as a
hook parameter (:class:`ForceFieldBuilder`, protocol defined HERE — this
module owns the contract, tools fills it).  The default builder is the
openmm-only :class:`PlainForceFieldBuilder` (no GAFF, no rename-after-match)
unless ``neomd2.tools.antechamber`` is importable and exposes a builder;
``neomd2.tools`` is imported lazily inside the default paths and its absence
degrades to the plain builder (or a clear error when ligand parameterization
is actually required).
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as etree
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .errors import ConfigValueError
from .plan import Plan

__all__ = [
    "SystemBundle",
    "ForceFieldBuilder",
    "PlainForceFieldBuilder",
    "prepare_system",
    "load_complex",
    "system_from_gromacs",
    "system_from_amber",
    "custom_bonds",
    "custom_addH",
    "sys_params_from_config",
]


# ---------------------------------------------------------------------------
# lazy engine imports (both halves of the module stay importable without them)
# ---------------------------------------------------------------------------


def _openmm():
    """(openmm, app, unit, XmlSerializer) with a clear error when absent."""
    try:
        import openmm
        from openmm import XmlSerializer, app, unit
    except ImportError as error:  # pragma: no cover - openmm is a project dep
        raise ImportError(
            "this neomd2.system workflow requires openmm "
            "(system preparation lives at the openmm boundary); "
            "SystemBundle.from_plan alone does not"
        ) from error
    return openmm, app, unit, XmlSerializer


# ---------------------------------------------------------------------------
# SystemBundle — the kernel-agnostic description (no openmm beyond this line
# until the workflow section)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemBundle:
    """Everything about a prepared system except the physics kernel.

    Fields
    ------
    topology_file:
        path of the .pdb/.pdbx carrying topology + positions.
    system_xml:
        path of the serialized openmm System (deserialized by the kernel).
    ligands:
        list of openff Molecule objects, or None when no ligands are bound.
    templates:
        ligand template identifiers (ffxml paths from a plan, or the template
        names the builder used during preparation).
    modifications:
        the modification IR — ``{"barostat": raw dict | None,
        "particle_masses": {int index: float dalton} | None}``.  The barostat
        dict is kept RAW (v1 ``NeoSystem.add_barostat`` defaults and the
        seed/temperature augmentation happen in ``run.compile`` -> KernelSpec,
        never here).
    """

    topology_file: str | None = None
    system_xml: str | None = None
    ligands: list | None = None
    templates: list[str] | None = None
    modifications: dict | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_plan(cls, plan) -> "SystemBundle":
        """Validate the plan's input_files and load the ligand molecules.

        * ``complex`` must exist and end in ``.pdb``/``.pdbx`` (the v1
          ``load_complex`` suffix check and message);
        * ``system`` must exist;
        * ``ligands`` (optional) is a json list of openff molecule dicts —
          openff.toolkit is imported lazily inside the loader because
          ligands are optional;
        * ``templates`` comes comma-split from ``plan.derived``;
        * ``barostat``/``system_modification`` are extracted into the
          modification IR (see module docstring).

        ``plan`` may be a :class:`~neomd2.plan.Plan` or a plain plan dict
        (validated through ``Plan.from_dict``).
        """
        if isinstance(plan, Plan):
            resolved = plan
        elif isinstance(plan, Mapping):
            resolved = Plan.from_dict(dict(plan))
        else:
            raise TypeError(
                f"SystemBundle.from_plan expects a Plan or a plan mapping, "
                f"got {type(plan).__name__}")

        input_files = resolved.input_files

        complex_path = input_files.get("complex")
        if complex_path is None:
            raise ConfigValueError(
                "input_files.complex is required to describe a system",
                key="complex")
        _check_complex_suffix(complex_path)
        _check_exists(complex_path, "input_files.complex")

        system_path = input_files.get("system")
        if system_path is None:
            raise ConfigValueError(
                "input_files.system is required to describe a system",
                key="system")
        _check_exists(system_path, "input_files.system")

        ligand_path = input_files.get("ligands")
        ligands = _load_ligands(ligand_path) if ligand_path else None

        return cls(
            topology_file=str(complex_path),
            system_xml=str(system_path),
            ligands=ligands,
            templates=getattr(resolved, "templates", None),
            modifications=_extract_modifications(resolved.raw),
        )

    # -- reporting -----------------------------------------------------------

    def describe(self) -> str:
        """A compact multi-line summary for logs and manifests."""
        ligands = self.ligands
        if ligands:
            names = ", ".join(
                getattr(mol, "name", "") or f"ligand{i + 1}"
                for i, mol in enumerate(ligands))
            ligand_line = f"ligands: {len(ligands)} ({names})"
        else:
            ligand_line = "ligands: none"
        modifications = self.modifications or {}
        barostat = modifications.get("barostat")
        masses = modifications.get("particle_masses")
        parts = []
        if barostat:
            items = ", ".join(
                f"{key}={value!r}" for key, value in sorted(barostat.items()))
            parts.append(f"barostat({items})")
        if masses:
            parts.append(f"{len(masses)} particle-mass overrides")
        mod_line = "modifications: " + ("; ".join(parts) if parts else "none")
        templates = (
            f"templates: {', '.join(self.templates)}" if self.templates
            else "templates: none")
        return "\n".join((
            "SystemBundle:",
            f"  topology: {self.topology_file}",
            f"  system:   {self.system_xml}",
            f"  {ligand_line}",
            f"  {templates}",
            f"  {mod_line}",
        ))


# ---------------------------------------------------------------------------
# loading / validation helpers (openmm-free)
# ---------------------------------------------------------------------------


def _check_complex_suffix(complex_path: str) -> None:
    """The v1 ``load_complex`` suffix gate, message verbatim."""
    if not (complex_path.endswith(".pdb") or complex_path.endswith(".pdbx")):
        raise ConfigValueError(
            "In config.input_files.complex, unrecognized file type:{}".format(
                complex_path),
            key="complex",
            value=complex_path)


def _check_exists(path: str, key: str) -> None:
    if not os.path.isfile(path):
        raise ConfigValueError(
            f"{key} file not found: {path}", key=key, value=path)


def _load_ligands(ligand_path: str) -> list:
    """Load the ligand json (a list of openff molecule dicts).

    openff.toolkit is imported lazily HERE: ligands are optional, and the
    rest of this module works without the toolkit.
    """
    _check_exists(ligand_path, "input_files.ligands")
    try:
        with open(ligand_path, "r") as handle:
            ligands_json = json.load(handle)
    except json.JSONDecodeError as error:
        raise ConfigValueError(
            f"input_files.ligands is not valid json: {error}",
            key="ligands", value=ligand_path) from error
    if not isinstance(ligands_json, list):
        raise ConfigValueError(
            "input_files.ligands must be a json LIST of openff molecule "
            f"dicts, got {type(ligands_json).__name__}",
            key="ligands", value=ligand_path)
    try:
        from openff.toolkit.topology import Molecule as openff_Molecule
    except ImportError as error:
        raise ConfigValueError(
            "input_files.ligands requires openff-toolkit, which is not "
            "importable in this environment (ligands are optional; remove "
            "the key to prepare a protein-only system)",
            key="ligands", value=ligand_path) from error
    return [
        openff_Molecule.from_json(json.dumps(liginfo))
        for liginfo in ligands_json
    ]


def _normalize_particle_masses(system_modification) -> dict[int, float] | None:
    """``{particle index: mass in dalton}`` from raw ``system_modification``.

    Verbatim port of v1 ``neosystem.py:74-78`` semantics (a mapping of
    ``{index: {"mass": value}}``; entries without a "mass" key are ignored),
    accepting BOTH spellings the plan schema allows — the v1 mapping and the
    list-of-dicts ``[{"index": i, "mass": m}, ...]`` (normalized exactly like
    ``run._particle_masses`` compiles into the KernelSpec).
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


def _extract_modifications(source: Mapping) -> dict:
    """The modification IR from a plan's RAW view (or a prepare config).

    The barostat dict stays RAW — the temperature/seed augmentation v1 did
    in ``add_barostat`` (``config.get("temperature", 298)`` /
    ``setRandomNumberSeed(config.seed)``) is ``run.compile``'s job, so the
    same source of truth feeds both the bundle and the KernelSpec.
    """
    if not isinstance(source, Mapping):  # pragma: no cover - defensive
        raise TypeError(
            f"modification source must be a mapping, "
            f"got {type(source).__name__}")
    barostat = source.get("barostat")
    if barostat is not None:
        barostat = dict(barostat)
    return {
        "barostat": barostat,
        "particle_masses": _normalize_particle_masses(
            source.get("system_modification")),
    }


def _templates_value(value) -> list[str] | None:
    """Comma-split a templates string (plans already did this in derive)."""
    if not value:
        return None
    if isinstance(value, str):
        return value.split(",")
    return [str(item) for item in value]


# ---------------------------------------------------------------------------
# v1 structure loading (openmm boundary — lazy imports)
# ---------------------------------------------------------------------------


def load_complex(complex_path):
    """Port of v1 ``io/system_loader.load_complex`` (PDB/PDBxFile)."""
    _check_complex_suffix(complex_path)
    _, app, _, _ = _openmm()
    if complex_path.endswith(".pdb"):
        return app.PDBFile(complex_path)
    return app.PDBxFile(complex_path)


# ---------------------------------------------------------------------------
# custom residue knowledge (ports of bin/prepare_openmm_system.py helpers;
# xml.etree only — no openmm import needed by custom_bonds itself)
# ---------------------------------------------------------------------------


def custom_bonds(top, pos, custom_config):
    """Port of v1 ``custom_bonds``: teach a topology about custom residues.

    ``custom_config`` maps resname -> ``{"bonds_from_ffxml": path}`` and/or
    ``{"custom_bonds": [[atom1, atom2], ...]}``.  A residue already known to
    the topology's standard bonds is an error (v1 message verbatim).
    """
    for resname, res_bonds in custom_config.items():
        if resname not in top._standardBonds:
            bonds = []
            print(f"res {resname} not in top._standardBonds,will add bonds")
            top._standardBonds[resname] = bonds
        else:
            raise ValueError(
                f"res {resname} found in top._standardBonds,cannot add bonds"
            )
        if res_bonds.get("bonds_from_ffxml"):
            tree = etree.parse(res_bonds["bonds_from_ffxml"])
            find_res = 0
            if tree.getroot().find("Residues") is not None:
                for residue in tree.getroot().find("Residues").findall("Residue"):
                    if residue.attrib["name"] == resname:
                        find_res = 1
                        for bond in residue.findall("Bond"):
                            bonds.append(
                                (bond.attrib["atomName1"], bond.attrib["atomName2"])
                            )
                        break
            if not find_res:
                raise ValueError(
                    'Cannot find info of residue "{}" in file "{}"'.format(
                        resname, res_bonds["bonds_from_ffxml"]
                    )
                )
        if res_bonds.get("custom_bonds"):
            for bond in res_bonds["custom_bonds"]:
                bonds.append((bond[0], bond[1]))
    top._bonds = []
    top.createStandardBonds()
    top.createDisulfideBonds(pos)


def custom_addH(modeller, forcefield, custom_config):
    """Port of v1 ``custom_addH``: teach modeller.addHydrogens about custom
    residues, reading the H-parent graph from an ffxml.

    ``forcefield`` may be a plain openmm ForceField or a ComplexForceField-
    shaped wrapper (anything with a ``.forcefield`` attribute holding the
    openmm ForceField whose ``_atomTypes`` carry the elements).
    """
    infinity = float("Inf")
    inner = getattr(forcefield, "forcefield", forcefield)
    for resname, res_addH in custom_config.items():
        data = modeller._ResidueData(resname)
        data.variants.append(resname)
        modeller._residueHydrogens[resname] = data
        if res_addH.get("H_from_ffxml"):
            tree = etree.parse(res_addH["H_from_ffxml"])
            find_res = 0
            if tree.getroot().find("Residues") is not None:
                for residue in tree.getroot().find("Residues").findall("Residue"):
                    if residue.attrib["name"] == resname:
                        find_res = 1
                        H_parents = {}
                        for atom in residue.findall("Atom"):
                            at_type = atom.attrib["type"]
                            if (
                                inner._atomTypes[at_type].element.symbol
                                == "H"
                            ):
                                H_parents[atom.attrib["name"]] = None
                        for bond in residue.findall("Bond"):
                            at1 = bond.attrib["atomName1"]
                            at2 = bond.attrib["atomName2"]
                            if at1 in H_parents:
                                H_parents[at1] = at2
                            elif at2 in H_parents:
                                H_parents[at2] = at1
                        break
            if not find_res:
                raise ValueError(
                    'Cannot find info of residue "{}" in file "{}"'.format(
                        resname, res_addH["H_from_ffxml"]
                    )
                )
            for hydrogen, parent in H_parents.items():
                maxph = infinity
                atomVariants = None
                terminal = None
                data.hydrogens.append(
                    modeller._Hydrogen(hydrogen, parent, maxph, atomVariants, terminal)
                )


def sys_params_from_config(sys_config):
    """Port of v1 ``ComplexForceField.sys_params_from_config`` (defaults are
    physics: constraints HBonds, nonbonded PME, cutoff 1.0 nm, rigid water,
    no CMMotion removal, hydrogenMass 4 amu)."""
    _, app, unit, _ = _openmm()
    if sys_config is None:
        sys_config = {}
    sys_config = dict(sys_config)
    sys_args = {}
    sys_config["constraints"] = sys_config.get("constraints", "HBonds")
    sys_config["nonbonded_method"] = sys_config.get("nonbonded_method", "pme")
    sys_args["nonbondedCutoff"] = (
        sys_config.get("nonbondedCutoff", 1.0) * unit.nanometers
    )
    sys_args["rigidWater"] = sys_config.get("rigidWater", True)
    sys_args["removeCMMotion"] = sys_config.get("removeCMMotion", False)
    sys_args["hydrogenMass"] = sys_config.get("hydrogenMass", 4) * unit.amu
    if sys_config.get("constraints") == "HBonds":
        sys_args["constraints"] = app.HBonds
    if sys_config.get("nonbonded_method") == "pme":
        sys_args["nonbondedMethod"] = app.PME
    return sys_args


# ---------------------------------------------------------------------------
# the tools seam: ForceFieldBuilder (protocol owned HERE, filled by tools/)
# ---------------------------------------------------------------------------


@runtime_checkable
class ForceFieldBuilder(Protocol):
    """The parameterization seam between the prepare workflow and tools/.

    The workflow (this module) owns topology assembly — modeller, hydrogens,
    solvent, box, centering — and calls the builder for everything that needs
    forcefield knowledge.  The tools layer (``neomd2.tools.antechamber``;
    ComplexForceField + GAFFTemplateGenerator knowledge) implements:

    ``build(topology, positions, ligands, ff_kwargs, sys_kwargs)``
        Build the final openmm System from the FINAL topology (post-H,
        post-solvent) and positions (openmm Quantity).  ``ligands`` is the
        list of openff Molecules placed by the workflow (or None);
        ``ff_kwargs`` is the ``ff_setting`` dict (base_ff / water_model /
        additional_forcefield_xml_path / rename_by_template / templates);
        ``sys_kwargs`` is the already unit-converted createSystem kwargs from
        :func:`sys_params_from_config`.  Returns
        ``(openmm System, used_ligand_templates)`` where the second element
        lists the ligand template identifiers the builder registered/used
        (empty for a protein-only build).

    ``openmm_forcefield(ff_kwargs, ligands=None)``
        The openmm ForceField object (ligand/GAFF template generators
        registered when ligands are given) used by the mid-workflow steps
        that need one: ``modeller.addSolvent`` and :func:`custom_addH`'s
        atom-element lookup.  It must expose ``createSystem`` (a plain
        ForceField or a ComplexForceField-shaped wrapper both do).
    """

    def build(self, topology, positions, ligands, ff_kwargs, sys_kwargs):
        ...

    def openmm_forcefield(self, ff_kwargs, ligands=None):
        ...


@dataclass
class PlainForceFieldBuilder:
    """openmm-only :class:`ForceFieldBuilder` — the no-tools fallback.

    Builds ``openmm.app.ForceField(base_ff, water_model, *additional xml)``
    (v1 ComplexForceField defaults) and calls ``createSystem``.  Limitations
    versus the tools layer, by construction:

    * no rename-after-match template matching (v1 ``rename_by_template``);
    * ligands require a GAFF generator hook (``gaff_factory``) — the default
      hook lazily imports ``neomd2.tools.antechamber`` and fails with a
      clear message when that layer is absent.
    """

    gaff_factory: Callable[[], Any] | None = None
    _cache: tuple | None = field(default=None, init=False, repr=False,
                                 compare=False)
    #: template identifiers registered for the ligands of the current build
    used_ligand_templates: list = field(default_factory=list, repr=False)

    def openmm_forcefield(self, ff_kwargs, ligands=None):
        _, app, _, _ = _openmm()
        ff_kwargs = dict(ff_kwargs or {})
        key = (
            ff_kwargs.get("base_ff"), ff_kwargs.get("water_model"),
            tuple(ff_kwargs.get("additional_forcefield_xml_path") or ()),
            None if ligands is None else tuple(id(mol) for mol in ligands),
        )
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        # v1 ComplexForceField.__init__ defaults, verbatim
        forcefield = app.ForceField(
            ff_kwargs.get("base_ff", "amber/protein.ff14SB.xml"),
            ff_kwargs.get("water_model", "amber/tip3p_standard.xml"),
        )
        additional = ff_kwargs.get("additional_forcefield_xml_path")
        if additional is not None:
            if isinstance(additional, str):
                additional = [additional]
            for _xml_path in additional:
                if os.path.exists(_xml_path):
                    forcefield.loadFile(_xml_path)
        used_templates: list[str] = []
        if ligands:
            factory = self.gaff_factory or _default_gaff_factory
            generator = factory()
            for ligand_mol in ligands:
                generator.add_molecules(ligand_mol)
            forcefield.registerTemplateGenerator(generator.generator)
            used_templates = [
                (getattr(mol, "name", "") or f"ligand{i + 1}")
                for i, mol in enumerate(ligands)
            ]
        self._cache = (key, forcefield)
        self.used_ligand_templates = used_templates
        return forcefield

    def build(self, topology, positions, ligands, ff_kwargs, sys_kwargs):
        forcefield = self.openmm_forcefield(ff_kwargs, ligands)
        return (forcefield.createSystem(topology, **sys_kwargs),
                list(self.used_ligand_templates))


def _default_gaff_factory():
    """GAFF generator factory from the tools layer (lazy, clear failure).

    Returns an INSTANCE (v1 ``GAFFTemplateGenerator`` shape): the plain
    builder calls ``add_molecules(mol)`` on the result and registers its
    BOUND ``.generator`` callback with the openmm ForceField — handing back
    the CLASS would register the unbound function and explode with a
    TypeError at residue matching, so classes are instantiated here
    (zero-arg construction; the tools generator's defaults apply).
    """
    try:
        import neomd2.tools.antechamber as _tools
    except ImportError as error:
        raise ConfigValueError(
            "parameterizing ligands requires the GAFF tools layer "
            "(neomd2.tools.antechamber), which is not importable yet: "
            f"{error}; pass prepare_system(..., gaff=<factory>) or use a "
            "tools-backed ForceFieldBuilder via forcefield="
        ) from error
    for attr in ("GAFFTemplateGenerator", "default_gaff_generator"):
        candidate = getattr(_tools, attr, None)
        if candidate is not None:
            return candidate() if isinstance(candidate, type) else candidate
    raise ConfigValueError(
        "neomd2.tools.antechamber does not expose GAFFTemplateGenerator "
        "or default_gaff_generator; the seam system.py expects is a "
        "zero-arg factory returning an object with add_molecules(mol) "
        "and a .generator attribute (v1 GAFFTemplateGenerator shape)")


def _default_forcefield_builder(gaff=None):
    """The default builder: the tools layer when present, else the plain one."""
    try:
        import neomd2.tools.antechamber as _tools
    except ImportError:
        return PlainForceFieldBuilder(gaff_factory=gaff)
    for attr in ("ForceFieldBuilder", "default_forcefield_builder"):
        candidate = getattr(_tools, attr, None)
        if candidate is None:
            continue
        if isinstance(candidate, type):
            try:
                return candidate(gaff_factory=gaff)
            except TypeError:
                return candidate()
        return candidate
    return PlainForceFieldBuilder(gaff_factory=gaff)


# ---------------------------------------------------------------------------
# ligand loading for the prepare workflow (v1 builder/ligand.py essentials;
# the full ligand workflow — smiles validation, charge assignment — lives in
# neomd2.tools.ligand, plan §5 item 2.6)
# ---------------------------------------------------------------------------


def _ligand_from_path(ligand_path: str):
    """Port of v1 ``Ligand.from_path`` (rdkit file -> openff Molecule)."""
    try:
        from rdkit import Chem
        from openff.toolkit.topology import Molecule as openff_Molecule
    except ImportError as error:
        raise ConfigValueError(
            f"loading ligand {ligand_path!r} requires rdkit + openff-toolkit "
            f"({error})", key="ligands", value=ligand_path) from error
    if ligand_path.endswith(".pdb"):
        rdkitmolh = Chem.MolFromPDBFile(ligand_path, removeHs=False)
    elif ligand_path.endswith(".sdf"):
        supp = Chem.ForwardSDMolSupplier(ligand_path, removeHs=False)
        rdkitmolh = next(supp)
    elif ligand_path.endswith(".mol2"):
        rdkitmolh = Chem.MolFromMol2File(ligand_path, removeHs=False)
    elif ligand_path.endswith(".mol"):
        rdkitmolh = Chem.MolFromMolFile(ligand_path, removeHs=False)
    else:
        raise ConfigValueError(
            f"unsupported ligand file type: {ligand_path} "
            "(expected .pdb/.sdf/.mol2/.mol)",
            key="ligands", value=ligand_path)
    if rdkitmolh is None:
        raise ConfigValueError(
            f"could not parse ligand file: {ligand_path}",
            key="ligands", value=ligand_path)
    if os.path.splitext(ligand_path)[1] == ".pdb":
        Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
    return openff_Molecule.from_rdkit(rdkitmolh, hydrogens_are_explicit=True)


def _ligands_from_config(ligands_config):
    """Ligand molecules from the prepare config's ``ligands`` section.

    Two accepted forms:

    * a sequence of openff Molecule objects (the prepared form the ligand
      workflow / tools layer produces);
    * the v1 mapping ``{name: {"path": ..., "resname": ...}}`` — and when an
      entry carries the ligand-workflow keys (``smiles`` /
      ``partial_charges``), the WHOLE mapping loads through
      ``neomd2.tools.ligand.ligands_from_config`` (plan §5 item 2.6): v1's
      SMILES graph validation, charge-file reading and ``template_ffxml``
      capture, with ``.molecule`` extracted so the workflow below keeps
      handling plain openff Molecules.
    """
    if not ligands_config:
        return None
    if isinstance(ligands_config, Mapping):
        mols = []
        if any(
            {"smiles", "partial_charges"} & set(info)
            for info in ligands_config.values()
            if isinstance(info, Mapping)
        ):
            try:
                from neomd2.tools.ligand import (
                    ligands_from_config as _ligand_workflow,
                )
            except ImportError as error:
                raise ConfigValueError(
                    "the ligand workflow (neomd2.tools.ligand) is not "
                    f"importable: {error}",
                    key="ligands") from error
            return [ligand.molecule
                    for ligand in _ligand_workflow(ligands_config)]
        for ligname, lig_info in ligands_config.items():
            mol = _ligand_from_path(lig_info["path"])
            if lig_info.get("resname"):
                mol.name = lig_info["resname"]
            elif mol.name == "":
                mol.name = "LIG"
            mols.append(mol)
        return mols
    return list(ligands_config)


# ---------------------------------------------------------------------------
# from_gromacs / from_amber branches (public loaders so callers and tests
# can substitute them)
# ---------------------------------------------------------------------------


def system_from_gromacs(config: Mapping):
    """Port of v1 ``prepare_system``'s from_gromacs branch.

    Returns (topology, positions, system, ligands=None).
    """
    _, app, _, _ = _openmm()
    gro = app.GromacsGroFile(config.get("gro"))
    _top = app.GromacsTopFile(
        config.get("top"),
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=config.get("ff_path"),
    )
    sys_args = sys_params_from_config(None)
    system = _top.createSystem(**sys_args)
    return _top.topology, gro.positions, system, None


def system_from_amber(config: Mapping):
    """Port of v1 ``prepare_system``'s from_amber branch.

    Returns (topology, positions, system, ligands=None).
    """
    _, app, _, _ = _openmm()
    coord = app.AmberInpcrdFile(config.get("inpcrd"))
    _top = app.AmberPrmtopFile(config.get("prmtop"))
    sys_args = sys_params_from_config(None)
    system = _top.createSystem(**sys_args)
    return _top.topology, coord.positions, system, None


# ---------------------------------------------------------------------------
# make_system — the orchestration port
# ---------------------------------------------------------------------------


def _make_system(
    protein_config, ligands_config, forcefield_kwargs, sys_params,
    additional_config, builder,
):
    """Port of v1 ``make_system`` (see module docstring for the seams)."""
    _, app, unit, _ = _openmm()

    protein = None
    ligands = None
    modeller = None
    ligands_pos_opmm_unit = None

    # if system without protein, no box_vectors/modeller got, then get them
    # from ligands
    if protein_config:
        protein = load_complex(protein_config["path"])
        if protein_config.get("custom_res_bonds"):
            custom_bonds(
                protein.topology, protein.positions, protein_config["custom_res_bonds"]
            )
        modeller = app.Modeller(protein.topology, protein.positions)
    if ligands_config:
        ligands = _ligands_from_config(ligands_config)
        if ligands is not None:
            for lig_i, ligand in enumerate(ligands):
                if ligand.conformers is None or len(ligand.conformers) == 0:
                    raise ConfigValueError(
                        f"ligand {lig_i} has no conformers; a placed "
                        "conformer is required to build the system",
                        key="ligands")
            ligands_pos_opmm_unit = [
                unit.Quantity(ligand.conformers[0].magnitude, unit.angstrom)
                for ligand in ligands
            ]

    forcefield_kwargs = dict(forcefield_kwargs or {})

    if ligands:
        # give unique names for all atoms in the molecule, so that we don't
        # need to rename them when generating topology and template
        for ligand in ligands:
            ligand.generate_unique_atom_names(suffix="")
        for lig_i, ligand_mol in enumerate(ligands):
            if modeller is None:
                modeller = app.Modeller(
                    ligand_mol.to_topology().to_openmm(),
                    ligands_pos_opmm_unit[lig_i],
                )
            else:
                modeller.add(
                    ligand_mol.to_topology().to_openmm(),
                    ligands_pos_opmm_unit[lig_i],
                )
            _res = [res for res in modeller.topology.residues()][-1]
            _res.name = ligand_mol.name

    # if didn't get box_vectors from protein, get it from ligands
    if modeller.getTopology().getPeriodicBoxVectors():
        box_vectors = modeller.getTopology().getPeriodicBoxVectors()
    else:
        if ligands_pos_opmm_unit is None:
            raise ConfigValueError(
                "cannot determine periodic box vectors: the protein has no "
                "CRYST1 box and no ligand positions are available; give the "
                "complex a periodic box or provide ligands",
                key="protein")
        from openmm.app.internal.unitcell import computePeriodicBoxVectors

        _unit = unit.nanometers
        pos_list = [
            pos.value_in_unit(_unit) for pos in ligands_pos_opmm_unit
        ]
        pos_np = np.concatenate(pos_list, axis=0) * _unit
        box_size = max(pos_np.max(axis=0) - pos_np.min(axis=0)) + 2 * 1 * _unit
        _size = box_size.value_in_unit(_unit)
        _angle = 90 * np.pi / 180.0
        box_vectors = computePeriodicBoxVectors(
            _size, _size, _size, _angle, _angle, _angle
        )
        modeller.getTopology().setPeriodicBoxVectors(box_vectors)

    if additional_config is None:
        additional_config = {}
    additional_config = {
        "add_hydrogens": True,
        "add_solv_ions": True,
        "ion_Strength": 0.1,
        **additional_config,
    }

    # the openmm ForceField the mid-workflow steps need (lazily built so a
    # solvent-free, custom-addH-free preparation never constructs one)
    solvent_ff = None

    def _need_forcefield():
        nonlocal solvent_ff
        if solvent_ff is None:
            solvent_ff = builder.openmm_forcefield(forcefield_kwargs, ligands)
        return solvent_ff

    if protein_config:
        if protein_config.get("custom_res_addH"):
            custom_addH(modeller, _need_forcefield(), protein_config["custom_res_addH"])

    if additional_config.get("add_hydrogens"):
        res_ls = [res for res in modeller.topology.residues()]
        resname_ls = len(res_ls) * [None]
        if protein_config:
            if protein_config.get("custom_resname_dict"):
                for _resid_resname in protein_config.get("custom_resname_dict"):
                    for resid, resname in _resid_resname.items():
                        mismatch = 0
                        if (
                            resname in ["CYS", "CYX", "CYM"]
                            and res_ls[resid - 1].name != "CYS"
                        ):
                            mismatch = 1
                        elif (
                            resname in ["HID", "HIE", "HIP"]
                            and res_ls[resid - 1].name != "HIS"
                        ):
                            mismatch = 1
                        if mismatch:
                            raise ValueError(
                                f"residue {resid}:specified {resname} not is not valid name of residue {res_ls[resid-1].name}\n"
                            )
                        else:
                            resname_ls[resid - 1] = resname
        resvariant = modeller.addHydrogens(variants=resname_ls)
        for index, res in enumerate(modeller.topology.residues()):
            if resvariant[index] is None:
                continue
            print(
                "residue {} {} change from {} to {}\n".format(
                    res.id, res.chain.id, res.name, resvariant[index]
                )
            )
            res.name = resvariant[index]
    box_center_vec = 0.5 * box_vectors[0] + 0.5 * box_vectors[1] + 0.5 * box_vectors[2]
    move_vec = box_center_vec - modeller.positions.mean()
    for i in range(len(modeller.positions)):
        modeller.positions[i] += move_vec
    if additional_config.get("add_solv_ions"):
        modeller.addSolvent(
            _need_forcefield(),
            boxVectors=box_vectors,
            ionicStrength=additional_config.get("ion_Strength") * unit.molar,
        )
    sys_args = sys_params_from_config(sys_params)
    system, used_ligand_templates = builder.build(
        modeller.topology, modeller.positions, ligands,
        forcefield_kwargs, sys_args)
    return modeller, ligands, system, used_ligand_templates


# ---------------------------------------------------------------------------
# the prepare workflow
# ---------------------------------------------------------------------------


def prepare_system(config: Mapping, *, forcefield: ForceFieldBuilder | None = None,
                   gaff: Callable[[], Any] | None = None) -> SystemBundle:
    """Prepare a system and write it to ``config["output_dir"]``.

    Port of v1 ``bin/prepare_openmm_system.py::prepare_system`` (config is
    the same dict/Box shape v1's yaml produced; a path is NOT accepted —
    load yaml/json yourself, workflows take data).  Branches:

    * ``from_gromacs`` / ``from_amber``: pre-parameterized topologies
      (``system_from_gromacs`` / ``system_from_amber``);
    * otherwise ``make_system`` orchestration: protein (custom bonds) ->
      ligand placement -> box -> custom addH -> addHydrogens (variants +
      custom_resname_dict validation) -> centering -> addSolvent ->
      builder.build (createSystem), and writes ``solv.pdbx``,
      ``ligand.json`` (when ligands) and ``system.xml``.

    ``forcefield``: a :class:`ForceFieldBuilder` (the tools seam); default
    resolves to the tools-layer builder when importable, else the openmm-only
    :class:`PlainForceFieldBuilder`.  ``gaff``: GAFF generator factory used
    by the default path when ligands must be parameterized (lazily imported
    from ``neomd2.tools.antechamber``; clear error when absent).

    Returns the :class:`SystemBundle` pointing at the written files.
    """
    _, app, _, XmlSerializer = _openmm()
    if not isinstance(config, Mapping):
        raise ConfigValueError(
            f"prepare config must be a mapping, got {type(config).__name__}")
    config = dict(config)
    output_dir = config.get("output_dir")
    if not output_dir or not str(output_dir).strip():
        raise ConfigValueError(
            "prepare config requires output_dir (the directory the "
            "solv.pdbx / system.xml / ligand.json artifacts are written to)",
            key="output_dir")

    if config.get("from_gromacs"):
        top, pos, system, ligands = system_from_gromacs(config["from_gromacs"])
        used_ligand_templates = None
    elif config.get("from_amber"):
        top, pos, system, ligands = system_from_amber(config["from_amber"])
        used_ligand_templates = None
    else:
        builder = forcefield if forcefield is not None \
            else _default_forcefield_builder(gaff)
        if (forcefield is not None and gaff is not None
                and getattr(builder, "gaff_factory", None) is None):
            builder.gaff_factory = gaff
        modeller, ligands, system, used_ligand_templates = _make_system(
            config.get("protein"),
            config.get("ligands"),
            forcefield_kwargs=config.get("ff_setting", None),
            sys_params=config.get("system_params"),
            additional_config=config.get("additional"),
            builder=builder,
        )
        top = modeller.topology
        pos = modeller.positions

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ligand_path = os.path.join(output_dir, "ligand.json")
    solv_path = os.path.join(output_dir, "solv.pdbx")

    app.PDBxFile.writeFile(
        top, pos, open(solv_path, "w"), keepIds=config.get("output_keepid", False)
    )
    if not ligands is None:
        with open(ligand_path, "w") as f:
            ligands_json = json.dumps(
                [json.loads(ligand.to_json()) for ligand in ligands]
            )
            f.write(ligands_json)

    system_path = os.path.join(output_dir, "system.xml")
    with open(system_path, "w") as f:
        print("\n *********system done***********\n")
        f.write(XmlSerializer.serialize(system))

    return SystemBundle(
        topology_file=solv_path,
        system_xml=system_path,
        ligands=ligands,
        templates=(used_ligand_templates
                   if used_ligand_templates else _templates_value(
                       config.get("templates"))),
        modifications=_extract_modifications(config),
    )
