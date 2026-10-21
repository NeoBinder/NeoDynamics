"""
prepare — the system-preparation workflow
(:class:`~neomd.system.SystemBundle` keeps the kernel-agnostic data).

A boundary workflow, not a core-spine module: it imports openmm directly at
call time (lazy, via :func:`_openmm`).  Every openmm PRIVATE attribute it
needs lives in :mod:`neomd.openmm_privates` (version-pinned,
source-scanned) — nothing here touches an underscored openmm name.  The
parameterization seam is :class:`ForceFieldBuilder` (protocol defined HERE;
default :class:`PlainForceFieldBuilder`); the GAFF implementation is
``neomd_preparation.tools.antechamber``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from neomd.errors import ConfigValueError
from neomd.openmm_privates import (
    compute_periodic_box_vectors,
    custom_addH,
    custom_bonds,
)

if TYPE_CHECKING:  # runtime import is lazy (system.py re-exports this module)
    from neomd.system import SystemBundle

__all__ = [
    "ForceFieldBuilder",
    "PlainForceFieldBuilder",
    "prepare_system",
    "export_system_bundle",
    "verify_prepared_system",
    "load_complex",
    "system_from_gromacs",
    "system_from_amber",
    "custom_bonds",
    "custom_addH",
    "sys_params_from_config",
]


# ---------------------------------------------------------------------------
# lazy engine import (importing this module alone stays openmm-free)
# ---------------------------------------------------------------------------


def _openmm():
    """(openmm, app, unit, XmlSerializer) with a clear error when absent."""
    try:
        import openmm
        from openmm import XmlSerializer, app, unit
    except ImportError as error:  # pragma: no cover - openmm is a project dep
        raise ImportError(
            "this neomd prepare workflow requires openmm "
            "(system preparation lives at the openmm boundary)"
        ) from error
    return openmm, app, unit, XmlSerializer


# ---------------------------------------------------------------------------
# structure loading
# ---------------------------------------------------------------------------


def load_complex(complex_path):
    """Topology + positions from a complex coordinate file (PDB/PDBxFile)."""
    from neomd.system import _check_complex_suffix

    _check_complex_suffix(complex_path)
    _, app, _, _ = _openmm()
    if complex_path.endswith(".pdb"):
        return app.PDBFile(complex_path)
    return app.PDBxFile(complex_path)


def _reject_missing_residue_positions(source_path):
    """Stop when a source declares polymer residues without coordinates.

    PDB SEQRES and PDBx entity/atom label sequence metadata are evidence of
    omitted residues. Bare coordinate files have no such evidence; callers
    must supply a coordinate-complete revision after external gap modeling.
    """
    path = Path(source_path)
    if path.suffix.lower() == ".pdb":
        declared, counts, observed = {}, {}, {}
        with path.open() as stream:
            for line in stream:
                if line.startswith("SEQRES"):
                    chain = line[11:12]
                    declared.setdefault(chain, []).extend(line[19:70].split())
                    try:
                        counts[chain] = int(line[13:17])
                    except ValueError as error:
                        raise ConfigValueError(
                            "invalid SEQRES residue count; cannot verify coordinate coverage",
                            key="protein", source=str(path),
                        ) from error
                elif line.startswith(("ATOM  ", "HETATM")):
                    chain = line[21:22]
                    key = (line[22:26], line[26:27])
                    observed.setdefault(chain, {}).setdefault(key, line[17:20].strip())
        for chain, names in declared.items():
            rows = [name for name in observed.get(chain, {}).values() if name in names]
            if len(names) != counts[chain] or rows != names:
                raise ConfigValueError(
                    f"polymer chain {chain.strip() or '(blank)'} declares {len(names)} SEQRES "
                    f"residues but has {len(rows)} matching coordinate residues; "
                    "missing or mismatched residue positions require upstream modeling "
                    "and a new coordinate-complete source before prepare_system",
                    key="protein", source=str(path),
                )
        return
    if path.suffix.lower() != ".pdbx":
        return
    try:
        import gemmi
    except ImportError as error:
        raise ConfigValueError(
            "PDBx polymer-sequence coverage requires gemmi; install the engine "
            "environment before prepare_system", key="protein", source=str(path),
        ) from error

    def rows(block, category):
        table = block.find_mmcif_category(category)
        tags = [tag.split(".", 1)[1] for tag in table.tags]
        return [dict(zip(tags, list(row), strict=True)) for row in table]

    try:
        block = gemmi.cif.read_file(str(path)).sole_block()
        sequence_rows = rows(block, "_entity_poly_seq.")
        if not sequence_rows:
            return
        declared = {}
        for row in sequence_rows:
            declared.setdefault(row["entity_id"], set()).add(int(row["num"]))
        chain_entities = {row["id"]: row["entity_id"]
                          for row in rows(block, "_struct_asym.")}
        observed = {}
        for row in rows(block, "_atom_site."):
            chain = row["label_asym_id"]
            entity = row.get("label_entity_id", chain_entities.get(chain))
            if entity not in declared:
                continue
            chain_entities.setdefault(chain, entity)
            if chain_entities[chain] != entity:
                raise ValueError("polymer chain has inconsistent entity identity")
            label = row.get("label_seq_id")
            if label in (None, ".", "?"):
                raise ValueError(f"polymer chain {chain} has no label sequence mapping")
            observed.setdefault(chain, set()).add(int(label))
        if not any(entity in declared for entity in chain_entities.values()):
            raise ValueError("declared polymer entities have no mapped chain")
        for chain, entity in chain_entities.items():
            if entity not in declared:
                continue
            missing = sorted(declared[entity] - observed.get(chain, set()))
            extra = sorted(observed.get(chain, set()) - declared[entity])
            if extra:
                raise ConfigValueError(
                    f"polymer chain {chain} has coordinate labels outside its declared "
                    f"sequence: {extra[:12]}; supply a consistent source before prepare_system",
                    key="protein", source=str(path),
                )
            if missing:
                preview = ", ".join(map(str, missing[:12]))
                if len(missing) > 12:
                    preview += ", ..."
                raise ConfigValueError(
                    f"polymer chain {chain} has {len(missing)} declared residues without "
                    f"coordinate positions (sequence positions: {preview}); "
                    "model missing residues upstream and submit a new coordinate-complete "
                    "source before prepare_system",
                    key="protein", source=str(path),
                )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ConfigValueError(
            f"cannot verify PDBx polymer sequence and coordinate mapping: {error}",
            key="protein", source=str(path),
        ) from error


def _reject_unqualified_residue_links(topology, source_path):
    """Reject attached residue groups that this builder cannot parameterize jointly."""
    from neomd_preparation.source import STANDARD_RESIDUES

    peptide_residues = STANDARD_RESIDUES | {"SEP", "ACE", "NME"}
    for left, right in topology.bonds():
        first, second = left.residue, right.residue
        if first is second:
            continue
        peptide_link = (
            first.chain is second.chain
            and abs(first.index - second.index) == 1
            and first.name in peptide_residues
            and second.name in peptide_residues
            and {left.name, right.name} == {"C", "N"}
        )
        disulfide_link = (
            first.name in {"CYS", "CYX"}
            and second.name in {"CYS", "CYX"}
            and left.name == right.name == "SG"
        )
        if peptide_link or disulfide_link:
            continue
        endpoints = (
            f"{first.name} {first.chain.id}:{first.id}:{left.name}",
            f"{second.name} {second.chain.id}:{second.id}:{right.name}",
        )
        raise ConfigValueError(
            "covalently attached residues " + " -- ".join(endpoints)
            + " require a qualified joint parameter template; "
            "prepare_system cannot prepare this source graph",
            key="protein",
            source=str(source_path),
        )


def sys_params_from_config(sys_config):
    """createSystem kwargs from the ``system_params`` config (defaults are
    physics: constraints HBonds, nonbonded PME, cutoff 1.0 nm, rigid water,
    no CMMotion removal, hydrogenMass 3 amu)."""
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
    sys_args["hydrogenMass"] = sys_config.get("hydrogenMass", 3) * unit.amu
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
    forcefield knowledge.  The tools layer (``neomd_preparation.tools.antechamber``;
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

        Particle indices follow topology atom indices, as in OpenMM
        ``createSystem``. A builder must not silently permute physical
        particles. Extra particles require explicit System virtual sites.

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
    (the legacy defaults) and calls ``createSystem``.  Limitations
    versus the tools layer, by construction:

    * no rename-after-match template matching;
    * ligands require a GAFF generator hook (``gaff_factory``) — the default
      hook lazily imports ``neomd_preparation.tools.antechamber`` and fails with a
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
        # legacy ForceField defaults, verbatim (physics, not architecture)
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

    Returns an INSTANCE (GAFFTemplateGenerator shape): the plain
    builder calls ``add_molecules(mol)`` on the result and registers its
    BOUND ``.generator`` callback with the openmm ForceField — handing back
    the CLASS would register the unbound function and explode with a
    TypeError at residue matching, so classes are instantiated here
    (zero-arg construction; the tools generator's defaults apply).
    """
    try:
        import neomd_preparation.tools.antechamber as _tools
    except ImportError as error:
        raise ConfigValueError(
            "parameterizing ligands requires the GAFF tools layer "
            "(neomd_preparation.tools.antechamber), which is not importable yet: "
            f"{error}; pass prepare_system(..., gaff=<factory>) or use a "
            "tools-backed ForceFieldBuilder via forcefield="
        ) from error
    for attr in ("GAFFTemplateGenerator", "default_gaff_generator"):
        candidate = getattr(_tools, attr, None)
        if candidate is not None:
            return candidate() if isinstance(candidate, type) else candidate
    raise ConfigValueError(
        "neomd_preparation.tools.antechamber does not expose GAFFTemplateGenerator "
        "or default_gaff_generator; the seam prepare.py expects is a "
        "zero-arg factory returning an object with add_molecules(mol) "
        "and a .generator attribute (v1 GAFFTemplateGenerator shape)")


def _default_forcefield_builder(gaff=None):
    """The default builder: the tools layer when present, else the plain one."""
    try:
        import neomd_preparation.tools.antechamber as _tools
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
# ligand loading for the prepare workflow (the full ligand workflow —
# smiles validation, charge assignment — lives in neomd_preparation.tools.ligand)
# ---------------------------------------------------------------------------


def _ligand_from_path(ligand_path: str):
    """rdkit file -> openff Molecule."""
    try:
        from openff.toolkit.topology import Molecule as openff_Molecule
        from rdkit import Chem
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
    * the mapping ``{name: {"path": ..., "resname": ...}}`` — and when an
      entry carries the ligand-workflow keys (``smiles`` /
      ``partial_charges``), the WHOLE mapping loads through
      ``neomd_preparation.tools.ligand.ligands_from_config``: SMILES graph validation,
      charge-file reading and ``template_ffxml``
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
                from neomd_preparation.tools.ligand import (
                    ligands_from_config as _ligand_workflow,
                )
            except ImportError as error:
                raise ConfigValueError(
                    "the ligand workflow (neomd_preparation.tools.ligand) is not "
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
    """(topology, positions, system, ligands=None) from a GROMACS gro/top."""
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
    """(topology, positions, system, ligands=None) from AMBER prmtop/inpcrd."""
    _, app, _, _ = _openmm()
    coord = app.AmberInpcrdFile(config.get("inpcrd"))
    _top = app.AmberPrmtopFile(config.get("prmtop"))
    sys_args = sys_params_from_config(None)
    system = _top.createSystem(**sys_args)
    return _top.topology, coord.positions, system, None


# ---------------------------------------------------------------------------
# make_system — the orchestration
# ---------------------------------------------------------------------------


def _make_system(
    protein_config, ligands_config, forcefield_kwargs, sys_params,
    additional_config, builder, lineage=None,
):
    """Build the solvated system (see module docstring for the seams)."""
    _, app, unit, _ = _openmm()

    protein = None
    ligands = None
    modeller = None
    ligands_pos_opmm_unit = None

    # if system without protein, no box_vectors/modeller got, then get them
    # from ligands
    if protein_config:
        _reject_missing_residue_positions(protein_config["path"])
        protein = load_complex(protein_config["path"])
        _reject_unqualified_residue_links(protein.topology, protein_config["path"])
        if lineage is not None:
            if (lineage.descriptors.get("protein", {}).get("kind") == "canonical_structure"
                    and protein.getNumFrames() != 1):
                from neomd_preparation.bundle import _error, _raise

                _raise([_error("protein.kind", "canonical_structure requires a single model/frame")])
            _bind_source(lineage, protein.topology, protein.positions, input_key="protein",
                         kind="structure", path=protein_config["path"],
                         loader="openmm.PDBFile" if str(protein_config["path"]).endswith(".pdb")
                         else "openmm.PDBxFile")
        if protein_config.get("custom_res_bonds"):
            custom_bonds(
                protein.topology, protein.positions, protein_config["custom_res_bonds"]
            )
            _reject_unqualified_residue_links(protein.topology, protein_config["path"])
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
        ligand_source_bytes = ([ligand.to_json().encode() for ligand in ligands]
                               if lineage is not None else None)
        # give unique names for all atoms in the molecule, so that we don't
        # need to rename them when generating topology and template
        for ligand in ligands:
            ligand.generate_unique_atom_names(suffix="")
        for lig_i, ligand_mol in enumerate(ligands):
            ligand_topology = ligand_mol.to_topology().to_openmm()
            if lineage is not None:
                info = (list(ligands_config.values())[lig_i]
                        if isinstance(ligands_config, Mapping) else {})
                _bind_source(lineage, ligand_topology, ligands_pos_opmm_unit[lig_i],
                             input_key=f"ligand:{lig_i}", kind="ligand", path=info.get("path"),
                             data=ligand_source_bytes[lig_i], loader="openff.Molecule.to_openmm")
            if modeller is None:
                modeller = app.Modeller(
                    ligand_topology,
                    ligands_pos_opmm_unit[lig_i],
                )
            else:
                modeller.add(
                    ligand_topology,
                    ligands_pos_opmm_unit[lig_i],
                )
            _res = [res for res in modeller.topology.residues()][-1]
            _res.name = ligand_mol.name

    if lineage is not None:
        lineage.bind_assembled(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                               allow_copied_atoms=True)

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
        _unit = unit.nanometers
        pos_list = [
            pos.value_in_unit(_unit) for pos in ligands_pos_opmm_unit
        ]
        pos_np = np.concatenate(pos_list, axis=0) * _unit
        box_size = max(pos_np.max(axis=0) - pos_np.min(axis=0)) + 2 * 1 * _unit
        _size = box_size.value_in_unit(_unit)
        _angle = 90 * np.pi / 180.0
        box_vectors = compute_periodic_box_vectors(
            (_size, _size, _size), (_angle, _angle, _angle))
        modeller.getTopology().setPeriodicBoxVectors(box_vectors)

    if additional_config is None:
        additional_config = {}
    additional_config = {
        "add_hydrogens": True,
        "add_solv_ions": True,
        "ion_Strength": 0.1,
        "center_model": True,  # centering is optional
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
        if lineage is not None:
            lineage.observe(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                            operation="add_hydrogens")
            lineage.steps[-1]["residue_variants"] = resvariant
        for index, res in enumerate(modeller.topology.residues()):
            if resvariant[index] is None:
                continue
            print(
                "residue {} {} change from {} to {}\n".format(
                    res.id, res.chain.id, res.name, resvariant[index]
                )
            )
            res.name = resvariant[index]
    # gate the model centering on additional_config's
    # center_model (default on) — some workflows need the input frame kept
    if additional_config.get("center_model"):
        box_center_vec = 0.5 * box_vectors[0] + 0.5 * box_vectors[1] + 0.5 * box_vectors[2]
        move_vec = box_center_vec - modeller.positions.mean()
        for i in range(len(modeller.positions)):
            modeller.positions[i] += move_vec
        if lineage is not None:
            lineage.translate(modeller.positions.value_in_unit(unit.nanometer),
                              move_vec.value_in_unit(unit.nanometer))
    if additional_config.get("add_solv_ions"):
        modeller.addSolvent(
            _need_forcefield(),
            boxVectors=box_vectors,
            ionicStrength=additional_config.get("ion_Strength") * unit.molar,
        )
        if lineage is not None:
            lineage.observe(modeller.topology, modeller.positions.value_in_unit(unit.nanometer),
                            operation="add_solvent")
            lineage.steps[-1]["ionic_strength_molar"] = additional_config.get("ion_Strength")
    sys_args = sys_params_from_config(sys_params)
    system, used_ligand_templates = builder.build(
        modeller.topology, modeller.positions, ligands,
        forcefield_kwargs, sys_args)
    return modeller, ligands, system, used_ligand_templates


# ---------------------------------------------------------------------------
# the prepare workflow
# ---------------------------------------------------------------------------


def prepare_system(config: Mapping, *, forcefield: ForceFieldBuilder | None = None,
                   gaff: Callable[[], Any] | None = None,
                   prepared_export: Mapping | None = None) -> SystemBundle:
    """Prepare a system and write it to ``config["output_dir"]``.

    ``config`` is the prepared config dict (a path is NOT accepted —
    load yaml/json yourself, workflows take data).  Branches:

    * ``from_gromacs`` / ``from_amber``: pre-parameterized topologies
      (``system_from_gromacs`` / ``system_from_amber``);
    * otherwise ``make_system`` orchestration: protein (custom bonds) ->
      ligand placement -> box -> custom addH -> addHydrogens (variants +
      custom_resname_dict validation) -> centering (gated by
      ``additional_config["center_model"]``, default on) ->
      addSolvent -> builder.build (createSystem), and writes ``solv.pdbx``,
      ``ligand.json`` (when ligands) and ``system.xml``.

    ``forcefield``: a :class:`ForceFieldBuilder` (the tools seam); default
    resolves to the tools-layer builder when importable, else the openmm-only
    :class:`PlainForceFieldBuilder`.  ``gaff``: GAFF generator factory used
    by the default path when ligands must be parameterized (lazily imported
    from ``neomd_preparation.tools.antechamber``; clear error when absent).

    After writing the artifacts, the openmm-free QC pass
    (:func:`neomd.qc.check_prepared_system`) reads the trio back and leaves
    ``qc_report.json`` in ``output_dir`` (config key ``qc``; default soft
    mode reports without gating — strict mode raises
    :class:`~neomd.errors.StructureQualityError` after the report lands).

    ``prepared_export={}`` additionally exports an immutable scientific bundle;
    optional ``sources`` bind client UUIDs to verified loader input positions.
    The returned bundle's ``prepared_manifest`` names its manifest. Omitting
    this keyword preserves the existing preparation behavior and artifacts.
    """
    # lazy: system.py re-exports this module's names, so importing its
    # helpers at module level would be circular
    from neomd.system import SystemBundle, _extract_modifications, _templates_value

    _, app, _, XmlSerializer = _openmm()
    if not isinstance(config, Mapping):
        raise ConfigValueError(
            f"prepare config must be a mapping, got {type(config).__name__}")
    config = dict(config)
    from neomd_preparation.bundle import PreparationLineage

    lineage = None if prepared_export is None else PreparationLineage(prepared_export)
    output_dir = config.get("output_dir")
    if not output_dir or not str(output_dir).strip():
        raise ConfigValueError(
            "prepare config requires output_dir (the directory the "
            "solv.pdbx / system.xml / ligand.json artifacts are written to)",
            key="output_dir")

    if config.get("from_gromacs"):
        top, pos, system, ligands = system_from_gromacs(config["from_gromacs"])
        used_ligand_templates = None
        if lineage is not None:
            _bind_source(lineage, top, pos, input_key="gromacs", kind="gromacs",
                         path=config["from_gromacs"]["gro"],
                         loader="openmm.GromacsTopFile+GromacsGroFile")
    elif config.get("from_amber"):
        top, pos, system, ligands = system_from_amber(config["from_amber"])
        used_ligand_templates = None
        if lineage is not None:
            _bind_source(lineage, top, pos, input_key="amber", kind="amber",
                         path=config["from_amber"]["inpcrd"],
                         loader="openmm.AmberPrmtopFile+AmberInpcrdFile")
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
            lineage=lineage,
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
    if ligands is not None:
        with open(ligand_path, "w") as f:
            ligands_json = json.dumps(
                [json.loads(ligand.to_json()) for ligand in ligands]
            )
            f.write(ligands_json)

    system_path = os.path.join(output_dir, "system.xml")
    with open(system_path, "w") as f:
        print("\n *********system done***********\n")
        f.write(XmlSerializer.serialize(system))

    bundle = SystemBundle(
        topology_file=solv_path,
        system_xml=system_path,
        ligands=ligands,
        templates=(used_ligand_templates
                   if used_ligand_templates else _templates_value(
                       config.get("templates"))),
        modifications=_extract_modifications(config),
    )

    # QC hook: quality-check the artifacts just written — the
    # same files a downstream run consumes — and leave qc_report.json next
    # to them.  Default mode soft: raw preparation inputs routinely carry
    # fixable clashes (that is what minimization is for), so the report
    # documents rather than gates unless qc.mode says strict.
    from neomd import qc as _qc

    _qc.check_prepared_system(
        topology_file=solv_path,
        system_xml=system_path,
        ligand_json=ligand_path if ligands is not None else None,
        qc_config=config.get("qc"),
        output_dir=output_dir,
    )
    if lineage is not None:
        from dataclasses import replace

        manifest = _publish_prepared(lineage, bundle, top, pos, system, config,
                                     Path(output_dir) / "prepared")
        bundle = replace(bundle, prepared_manifest=manifest)
    return bundle


def _bind_source(lineage, topology, positions, *, input_key, kind, path=None,
                 data=None, loader):
    openmm, _, unit, _ = _openmm()
    version = openmm.__version__
    if loader.startswith("openff"):
        from importlib.metadata import version as package_version

        version = package_version("openff-toolkit")
    lineage.append_source(topology, positions.value_in_unit(unit.nanometer),
                          input_key=input_key, kind=kind, source_path=path, source_bytes=data,
                          loader={"name": loader, "version": version,
                                  "atom_order": "preparation_input"})


def _particle_inventory(system, positions, topology_particle_indices=None):
    """Export public System data and ask OpenMM to evaluate virtual positions.

    A force-free scratch System computes virtual sites without changing the
    prepared System, constructing its production integrator, or taking steps.
    Extra particles must be declared virtual sites; topology order is the
    documented OpenMM createSystem order unless the importer supplies a map.
    """
    openmm, _, unit, _ = _openmm()
    from neomd_preparation.bundle import _error, _raise

    count = system.getNumParticles()
    initial = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    mapping = list(range(len(initial))) if topology_particle_indices is None else list(topology_particle_indices)
    if (len(mapping) != len(initial) or len(set(mapping)) != len(mapping)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in mapping)):
        _raise([_error("topology_particle_indices", "must uniquely map each topology atom into the System")])
    coords = np.zeros((count, 3), dtype=np.float64)
    coords[mapping] = initial
    mapped_indices = set(mapping)
    inventory, clones = [], {}
    for i in range(count):
        item = {"mass_dalton": system.getParticleMass(i).value_in_unit(unit.dalton)}
        if system.isVirtualSite(i):
            site = system.getVirtualSite(i)
            parents = [site.getParticle(j) for j in range(site.getNumParticles())]
            desc = {"type": type(site).__name__, "parent_indices": parents}
            if isinstance(site, (openmm.TwoParticleAverageSite, openmm.ThreeParticleAverageSite)):
                weights = [site.getWeight(j) for j in range(site.getNumParticles())]
                desc["weights"] = weights
                clone = type(site)(*parents, *weights)
            elif isinstance(site, openmm.OutOfPlaneSite):
                weights = [site.getWeight12(), site.getWeight13(), site.getWeightCross()]
                desc.update(weights=weights, cross_weight_unit="1/nm")
                clone = openmm.OutOfPlaneSite(*parents, *weights)
            elif isinstance(site, openmm.LocalCoordinatesSite):
                origin, x, y = site.getOriginWeights(), site.getXWeights(), site.getYWeights()
                local = site.getLocalPosition()
                desc.update(origin_weights=list(origin), x_weights=list(x), y_weights=list(y),
                            local_position_nm=list(local.value_in_unit(unit.nanometer)))
                clone = openmm.LocalCoordinatesSite(parents, origin, x, y, local)
            else:
                _raise([_error(f"particles[{i}]", f"unsupported virtual-site class {type(site).__name__}")])
            item["virtual_site"] = desc
            clones[i] = clone
        elif i not in mapped_indices:
            _raise([_error(f"particles[{i}]", "non-virtual particle has no topology correspondence")])
        inventory.append(item)
    if clones:
        scratch = openmm.System()
        for i in range(count):
            scratch.addParticle(0 if i in clones else 1)
        for i, site in clones.items():
            scratch.setVirtualSite(i, site)
        integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
        context = openmm.Context(scratch, integrator, openmm.Platform.getPlatformByName("Reference"))
        try:
            context.setPositions(coords * unit.nanometer)
            context.computeVirtualSites()
            coords = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        finally:
            del context, integrator
    return inventory, coords


def _publish_prepared(lineage, bundle, topology, positions, system, config, directory,
                      topology_particle_indices=None):
    _, _, unit, _ = _openmm()
    from neomd_preparation.bundle import file_digest

    artifacts = {"system.xml": bundle.system_xml,
                 "topology" + Path(bundle.topology_file).suffix: bundle.topology_file}
    parent = Path(bundle.system_xml).parent
    for name in ("ligand.json", "qc_report.json"):
        if (parent / name).is_file():
            artifacts[name] = parent / name

    def normalize(value):
        if isinstance(value, Mapping):
            return {str(k): normalize(v) for k, v in value.items() if k != "output_dir"}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, (str, Path)):
            path = Path(value)
            try:
                if path.is_file():
                    name = f"inputs/{len(artifacts):04d}{path.suffix}"
                    artifacts[name] = path
                    return {"artifact": name, "sha256": file_digest(path)}
                if path.is_dir():
                    return {"external_directory_name": path.name,
                            "coverage": "effective parameters captured in system.xml; directory bytes not captured"}
            except OSError:
                pass
            return str(value)
        if hasattr(value, "to_json"):
            return json.loads(value.to_json())
        return value

    parameters = {"preparation": normalize(config), "system_modifications": bundle.modifications,
                  "system_scope": "prepared System before runtime Plan modifications and method biases",
                  "version_provenance": "export environment; imported files do not imply their original producer versions",
                  "qc_report_artifact": "qc_report.json" if "qc_report.json" in artifacts else None,
                  "particle_order_contract": "OpenMM topology order" if topology_particle_indices is None
                  else "explicit importer topology_particle_indices",
                  "system_constraints": system.getNumConstraints(),
                  "force_types": [type(force).__name__ for force in system.getForces()]}
    inventory, coords = _particle_inventory(system, positions, topology_particle_indices)
    box = system.getDefaultPeriodicBoxVectors()
    return lineage.publish(directory, topology=topology,
                           positions_nm=positions.value_in_unit(unit.nanometer), particles=inventory,
                           coordinates_nm=coords,
                           box_vectors_nm=[list(v.value_in_unit(unit.nanometer)) for v in box]
                           if system.usesPeriodicBoundaryConditions() else None,
                           artifacts=artifacts, parameters=parameters,
                           topology_particle_indices=topology_particle_indices)


def export_system_bundle(bundle, directory, *, sources=None, topology_particle_indices=None):
    """Export an existing prepared pair without inventing pre-preparation lineage.

    Source indices belong to this prepared topology, not an earlier molecule.
    Importers whose particle order differs from topology order must supply the
    explicit map. Returns a new SystemBundle with ``prepared_manifest`` set.
    """
    from dataclasses import replace

    from neomd_preparation.bundle import PreparationLineage

    _, _, _, serializer = _openmm()
    structure = load_complex(str(bundle.topology_file))
    system = serializer.deserialize(Path(bundle.system_xml).read_text())
    lineage = PreparationLineage({"sources": [] if sources is None else sources})
    _bind_source(lineage, structure.topology, structure.positions, input_key="prepared", kind="prepared",
                 path=bundle.topology_file, loader="openmm.PDBFile" if str(bundle.topology_file).endswith(".pdb")
                 else "openmm.PDBxFile")
    manifest = _publish_prepared(lineage, bundle, structure.topology, structure.positions,
                                 system, {}, directory, topology_particle_indices)
    return replace(bundle, prepared_manifest=manifest)


def verify_prepared_system(manifest_path, *, expected_sha256=None, profile=None,
                           trusted_topology_file=None, expected_system_constraints=None,
                           expected_force_types=None):
    """Verify actual System/topology correspondence after bundle byte validation.

    For untrusted intake the caller MUST approve the XML force/type vocabulary
    before calling: OpenMM deserializes it here. No plugin or service policy is
    implemented by this generic engine API. Identity is positional through the
    explicit topology map, never recovered from atom names. Coordinate checks
    allow only the prepared PDB/PDBx writer's decimal quantization (0.000051 /
    0.0000051 nm); masses/site parameters retain exact serialized float values.
    Returns the verified manifest; physical stability still needs a Context/QC
    gate and is not certified by this structural correspondence check.

    ``profile=None`` retains generic verification. The explicit
    ``canonical_protein_load_v1`` profile additionally verifies an unchanged
    single canonical protein load in v2, not preparation or forcefield fitness.
    ``canonical_protein_group_v1`` verifies only LOAD followed by deterministic
    protein/source-water grouping, not repair or assembly-report semantics.
    ``canonical_protein_repair_v1`` independently checks LOAD→GROUP→heavy repair
    consistency, not proof of backend execution or physical fitness.
    """
    from neomd_preparation.bundle import _error, _raise, read_prepared_bundle

    if profile not in (None, "canonical_protein_load_v1", "canonical_protein_group_v1",
                       "canonical_protein_repair_v1"):
        _raise([_error("profile", "unsupported prepared verification profile")])
    manifest = read_prepared_bundle(manifest_path, expected_sha256=expected_sha256)
    root = Path(manifest_path).resolve().parent
    names = [name for name in manifest["artifacts"] if name in {"topology.pdb", "topology.pdbx"}]
    if (len(names) != (0 if trusted_topology_file is not None else 1)
            or trusted_topology_file is not None
            and (profile is not None or manifest["schema_version"] != 2)
            or trusted_topology_file is not None
            and (type(expected_system_constraints) is not int
                 or not isinstance(expected_force_types, list))
            or "system.xml" not in manifest["artifacts"]):
        _raise([_error("artifacts", "exactly one topology and a System XML are required")])
    _, _, unit, serializer = _openmm()
    errors = []
    try:
        system = serializer.deserialize((root / "system.xml").read_text())
        structure = load_complex(str(trusted_topology_file or root / names[0]))
    except Exception as error:
        raise ValueError("prepared System or topology cannot be loaded") from error
    if system.getNumParticles() != manifest["particle_count"]:
        errors.append(_error("particle_count", "actual System particle count differs from manifest"))
    atoms = list(structure.topology.atoms())
    if len(atoms) != manifest["topology_atom_count"]:
        errors.append(_error("topology_atom_count", "actual topology atom count differs from manifest"))
    _raise(errors)  # Remaining row comparisons require matching dimensions.
    mapping = manifest["topology_particle_indices"]
    coords = np.fromfile(root / manifest["coordinates"]["artifact"], dtype="<f8").reshape((-1, 3))
    inventory, evaluated = _particle_inventory(system, coords[mapping] * unit.nanometer, mapping)
    for i, (actual, declared) in enumerate(zip(inventory, manifest["particles"])):
        if actual["mass_dalton"] != declared["mass_dalton"]:
            errors.append(_error(f"particles[{i}].mass_dalton", "actual System mass differs from manifest"))
        actual_site = actual.get("virtual_site")
        declared_site = declared.get("virtual_site")
        if declared_site is not None:
            declared_site = {key: value for key, value in declared_site.items() if key != "parent_ids"}
        if actual_site != declared_site:
            errors.append(_error(f"particles[{i}].virtual_site", "actual System virtual-site definition differs from manifest"))
    if not np.allclose(evaluated, coords, rtol=0, atol=1e-12):
        errors.append(_error("coordinates", "virtual-site positions disagree with the actual System"))
    loaded_coords = np.asarray(structure.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    tolerance = 0.000051 if names and names[0].endswith(".pdb") else 0.0000051
    if not np.allclose(loaded_coords, coords[mapping], rtol=0, atol=tolerance):
        errors.append(_error("topology_particle_indices", "topology coordinates do not match declared System rows within writer precision"))
    residues = {row["residue_id"]: row for row in manifest["residues"]}
    actual_residues = {}
    for i, (atom, declared) in enumerate(zip(atoms, manifest["atoms"])):
        element = None if atom.element is None else atom.element.symbol
        if element != declared["element"] or atom.name != declared["name"]:
            errors.append(_error(f"atoms[{i}]", "topology element/name differs at this declared row"))
        residue = residues[declared["residue_id"]]
        if atom.residue.name != residue["name"]:
            errors.append(_error(f"atoms[{i}].residue_id", "topology residue name differs at this declared row"))
        actual_residues.setdefault(atom.residue.index, set()).add(declared["residue_id"])
    if (any(len(ids) != 1 for ids in actual_residues.values())
            or len({next(iter(ids)) for ids in actual_residues.values()}) != len(actual_residues)):
        errors.append(_error("residues", "topology residue membership differs from manifest"))
    ids = [row["atom_id"] for row in manifest["atoms"]]
    actual_bonds = {frozenset((ids[a.index], ids[b.index])) for a, b in structure.topology.bonds()}
    declared_bonds = {frozenset(row["atom_ids"]) for row in manifest["bonds"]}
    if actual_bonds != declared_bonds:
        errors.append(_error("bonds", "topology bond endpoints differ from manifest"))
    parameters = manifest["parameters"]
    constraints = (expected_system_constraints if trusted_topology_file is not None
                   else parameters["system_constraints"])
    force_types = (expected_force_types if trusted_topology_file is not None
                   else parameters["force_types"])
    if system.getNumConstraints() != constraints:
        errors.append(_error("parameters.system_constraints", "actual constraint count differs from manifest"))
    if [type(force).__name__ for force in system.getForces()] != force_types:
        errors.append(_error("parameters.force_types", "actual force inventory differs from manifest"))
    box = manifest["box_vectors_nm"]
    actual_box = [list(v.value_in_unit(unit.nanometer)) for v in system.getDefaultPeriodicBoxVectors()]
    if system.usesPeriodicBoundaryConditions() != (box is not None) or (
            box is not None and not np.allclose(actual_box, box, rtol=0, atol=1e-12)):
        errors.append(_error("box_vectors_nm", "actual periodic System box differs from manifest"))
    _raise(errors)
    if profile == "canonical_protein_load_v1":
        from neomd_preparation.verify.load import verify_canonical_protein_load

        verify_canonical_protein_load(manifest, root, structure, coords)
    elif profile == "canonical_protein_group_v1":
        from neomd_preparation.verify.grouping import verify_canonical_protein_group

        verify_canonical_protein_group(manifest, root, structure, coords)
    elif profile == "canonical_protein_repair_v1":
        from neomd_preparation.verify.repair import verify_repair_stage

        verify_repair_stage(manifest, root, structure, coords)
    return manifest
