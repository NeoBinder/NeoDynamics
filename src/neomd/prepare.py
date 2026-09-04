"""prepare — the system-preparation WORKFLOW (system.py keeps the
kernel-agnostic :class:`~neomd.system.SystemBundle`, this module owns the
system-preparation orchestration).

This is a WORKFLOW, not a core-spine module, so it imports openmm directly
at call time (the "only kernel/openmm.py imports openmm in core"
rule refers to the spine; system preparation, like the openmm adapter,
lives at the openmm boundary).  The openmm import is lazy (via
:func:`_openmm`) so importing this module alone never drags the engine in.

Every openmm PRIVATE attribute the workflow needs lives in
:mod:`neomd.openmm_privates` (version-pinned, isolated, source-scanned) —
nothing below touches an underscored openmm name.

The tools seam: the heavy
parameterization knowledge (ComplexForceField, GAFF template generation,
rename-after-match) lives in ``neomd.tools.antechamber``;
:func:`prepare_system` takes that layer as a hook parameter
(:class:`ForceFieldBuilder`, protocol defined HERE).  The default builder
is the openmm-only :class:`PlainForceFieldBuilder` unless
``neomd.tools.antechamber`` is importable and exposes a builder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .errors import ConfigValueError
from .openmm_privates import (
    compute_periodic_box_vectors,
    custom_addH,
    custom_bonds,
)

if TYPE_CHECKING:  # runtime import is lazy (system.py re-exports this module)
    from .system import SystemBundle

__all__ = [
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
    from .system import _check_complex_suffix

    _check_complex_suffix(complex_path)
    _, app, _, _ = _openmm()
    if complex_path.endswith(".pdb"):
        return app.PDBFile(complex_path)
    return app.PDBxFile(complex_path)


def sys_params_from_config(sys_config):
    """createSystem kwargs from the ``system_params`` config (defaults are
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
    forcefield knowledge.  The tools layer (``neomd.tools.antechamber``;
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
    (the legacy defaults) and calls ``createSystem``.  Limitations
    versus the tools layer, by construction:

    * no rename-after-match template matching;
    * ligands require a GAFF generator hook (``gaff_factory``) — the default
      hook lazily imports ``neomd.tools.antechamber`` and fails with a
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
        import neomd.tools.antechamber as _tools
    except ImportError as error:
        raise ConfigValueError(
            "parameterizing ligands requires the GAFF tools layer "
            "(neomd.tools.antechamber), which is not importable yet: "
            f"{error}; pass prepare_system(..., gaff=<factory>) or use a "
            "tools-backed ForceFieldBuilder via forcefield="
        ) from error
    for attr in ("GAFFTemplateGenerator", "default_gaff_generator"):
        candidate = getattr(_tools, attr, None)
        if candidate is not None:
            return candidate() if isinstance(candidate, type) else candidate
    raise ConfigValueError(
        "neomd.tools.antechamber does not expose GAFFTemplateGenerator "
        "or default_gaff_generator; the seam prepare.py expects is a "
        "zero-arg factory returning an object with add_molecules(mol) "
        "and a .generator attribute (v1 GAFFTemplateGenerator shape)")


def _default_forcefield_builder(gaff=None):
    """The default builder: the tools layer when present, else the plain one."""
    try:
        import neomd.tools.antechamber as _tools
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
# smiles validation, charge assignment — lives in neomd.tools.ligand)
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
      ``neomd.tools.ligand.ligands_from_config``: SMILES graph validation,
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
                from neomd.tools.ligand import (
                    ligands_from_config as _ligand_workflow,
                )
            except ImportError as error:
                raise ConfigValueError(
                    "the ligand workflow (neomd.tools.ligand) is not "
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
    additional_config, builder,
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
    from ``neomd.tools.antechamber``; clear error when absent).

    After writing the artifacts, the openmm-free QC pass
    (:func:`neomd.qc.check_prepared_system`) reads the trio back and leaves
    ``qc_report.json`` in ``output_dir`` (config key ``qc``; default soft
    mode reports without gating — strict mode raises
    :class:`~neomd.errors.StructureQualityError` after the report lands).

    Returns the :class:`SystemBundle` pointing at the written files.
    """
    # lazy: system.py re-exports this module's names, so importing its
    # helpers at module level would be circular
    from .system import SystemBundle, _extract_modifications, _templates_value

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
    from . import qc as _qc

    _qc.check_prepared_system(
        topology_file=solv_path,
        system_xml=system_path,
        ligand_json=ligand_path if ligands is not None else None,
        qc_config=config.get("qc"),
        output_dir=output_dir,
    )
    return bundle
