"""
openmm_privates — the ONE home for openmm private-API usage (source-scanned).

Four usages with no public equivalent at the pinned release: custom bond
graph teaching (:func:`custom_bonds`), custom-residue hydrogen teaching in
``modeller.addHydrogens`` (:func:`custom_addH`), the ligand-extent fallback
box (:func:`compute_periodic_box_vectors`), retained-water ion allocation
(:func:`add_ions_preserving_water`). Pinned-version gate: the first
call checks ``openmm.__version__`` against :data:`PINNED_OPENMM_PREFIXES`
and raises :class:`~neomd.errors.UpstreamVersionError` outside openmm 8.6.x
— a loud refusal, never silent behavioral drift.  Re-pin by re-verifying the
usages against the new release and extending the prefix tuple.
"""

from __future__ import annotations

import xml.etree.ElementTree as etree
from copy import deepcopy
from inspect import signature
from math import isfinite

from .errors import ConfigValueError, PlanValidationErrors, UpstreamVersionError

__all__ = [
    "PINNED_OPENMM_PREFIXES",
    "assert_pinned_openmm",
    "custom_bonds",
    "custom_addH",
    "compute_periodic_box_vectors",
    "add_ions_preserving_water",
]

#: openmm minor-series prefixes whose private surface was verified
PINNED_OPENMM_PREFIXES: tuple[str, ...] = ("8.6",)

_checked: bool = False


def assert_pinned_openmm(openmm_module) -> None:
    """Raise :class:`UpstreamVersionError` unless the installed openmm is
    inside a pinned prefix range (checked once per process)."""
    global _checked
    if _checked:
        return
    version = str(getattr(openmm_module, "__version__", ""))
    if not any(version.startswith(prefix)
               for prefix in PINNED_OPENMM_PREFIXES):
        raise UpstreamVersionError(
            f"openmm {version} is outside the pinned private-API range "
            f"{PINNED_OPENMM_PREFIXES}; neomd's system preparation touches "
            f"openmm internals (Topology._standardBonds, "
            f"Modeller._ResidueData/_residueHydrogens/_Hydrogen, "
            f"ForceField._atomTypes, app.internal.unitcell) that were "
            f"verified on the pinned releases only — see "
            f"neomd/openmm_privates.py to re-pin after verification",
            value=version,
            candidates=list(PINNED_OPENMM_PREFIXES))
    _checked = True


def _openmm():
    try:
        import openmm
        from openmm import app
    except ImportError as error:  # pragma: no cover - openmm is a project dep
        raise ImportError(
            "this neomd preparation workflow requires openmm"
        ) from error
    assert_pinned_openmm(openmm)
    return app


# -- Topology._standardBonds / _bonds ------------------------------------------


def custom_bonds(top, pos, custom_config):
    """Teach a topology about custom residues.

    ``custom_config`` maps resname -> ``{"bonds_from_ffxml": path}`` and/or
    ``{"custom_bonds": [[atom1, atom2], ...]}``.  A residue already known to
    the topology's standard bonds is an error (message kept verbatim).

    Private surface: ``top._standardBonds`` (read/write), ``top._bonds``
    (write), then the public ``createStandardBonds`` /
    ``createDisulfideBonds`` rebuild.
    """
    _openmm()  # import + pin check before the first private attribute access
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


# -- Modeller._ResidueData / _residueHydrogens / _Hydrogen ---------------------


def custom_addH(modeller, forcefield, custom_config):
    """Teach modeller.addHydrogens about custom
    residues, reading the H-parent graph from an ffxml.

    ``forcefield`` may be a plain openmm ForceField or a ComplexForceField-
    shaped wrapper (anything with a ``.forcefield`` attribute holding the
    openmm ForceField whose ``_atomTypes`` carry the elements).

    Private surface: ``modeller._ResidueData``, ``modeller._residueHydrogens``,
    ``modeller._Hydrogen``, ``ForceField._atomTypes``.
    """
    _openmm()  # import + pin check before the first private attribute access
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


# -- app.internal.unitcell ------------------------------------------------------


def compute_periodic_box_vectors(size, angles):
    """``openmm.app.internal.unitcell.computePeriodicBoxVectors`` for the
    ligand-extent fallback box (the only internal-module import)."""
    from openmm.app.internal import unitcell

    _openmm()  # pin check first
    return unitcell.computePeriodicBoxVectors(*size, *angles)


def _ion_error(key, message):
    return ConfigValueError(message, key=f"ions.{key}")


def _ion_raise(errors):
    if errors:
        raise PlanValidationErrors(errors, footer="input modeller is unchanged")


def _ion_atom_keys(modeller, unit):
    atoms = list(modeller.topology.atoms())
    positions = modeller.positions.value_in_unit(unit.nanometer)
    if (len(atoms) != len(positions) or any(len(p) != 3 or not all(isfinite(v) for v in p)
                                           for p in positions)):
        _ion_raise([_ion_error("positions", "requires one finite coordinate triple per atom")])
    keys = [(a.element.symbol if a.element is not None else None, tuple(p))
            for a, p in zip(atoms, positions)]
    if len(set(keys)) != len(keys):
        _ion_raise([_ion_error("positions", "ambiguous atom correspondence; resolve coincident same-element atoms")])
    return dict(zip(atoms, keys))


def _ion_structure(modeller, keys, excluded):
    """Compare preserved ordering/grouping/metadata/bonds without label inference."""
    chains = []
    for chain in modeller.topology.chains():
        residues = []
        for residue in chain.residues():
            atoms = [(a.name, a.id, a.formalCharge, keys[a])
                     for a in residue.atoms() if a not in excluded]
            if atoms:
                residues.append((residue.name, residue.id, residue.insertionCode, atoms))
        if residues:
            chains.append((chain.id, residues))
    bonds = [(frozenset((keys[a], keys[b])), bond.order, str(bond.type))
             for bond in modeller.topology.bonds() for a, b in [bond]
             if a not in excluded and b not in excluded]
    return chains, bonds, modeller.topology.getPeriodicBoxVectors()


def add_ions_preserving_water(modeller, forcefield, *, retained_water_residues,
                             generated_water_residues, positive_ion, negative_ion,
                             ionic_strength_mol_per_l, neutralize):
    """Count all HOH waters but replace only explicitly generated waters.

    Supply a disjoint, complete partition using live, fully hydrogenated HOH
    residue objects. All scientific settings are required. The pinned OpenMM
    8.6 algorithm retains its 0.5 nm ion cutoff and rounded water/55.4 M count.
    UUID provenance belongs to the caller's lineage, not these topology labels.
    Success replaces topology/positions; any failure leaves the input unchanged.
    """
    import openmm
    from openmm import unit

    app = _openmm()
    version = str(openmm.__version__)
    if version.split(".")[:2] != ["8", "6"]:
        raise UpstreamVersionError(f"openmm {version} is outside pinned retained-water ion adapter 8.6.x")
    parameters = signature(app.Modeller._addIons).parameters
    if (tuple(parameters) != ("self", "forcefield", "numWaters", "replaceableMols", "ionCutoff",
                              "positiveIon", "negativeIon", "ionicStrength", "neutralize", "residueTemplates")
            or parameters["ionCutoff"].default != 0.5 * unit.nanometer):
        raise UpstreamVersionError("pinned OpenMM ion interface changed; requalify before preparation")
    errors = []
    if positive_ion != "Na+":
        errors.append(_ion_error("positive_ion", "this adapter is qualified for Na+ only"))
    if negative_ion != "Cl-":
        errors.append(_ion_error("negative_ion", "this adapter is qualified for Cl- only"))
    try:
        valid_salt = (type(ionic_strength_mol_per_l) in (int, float)
                      and isfinite(ionic_strength_mol_per_l) and ionic_strength_mol_per_l >= 0)
    except OverflowError:
        valid_salt = False
    if not valid_salt:
        errors.append(_ion_error("ionic_strength_mol_per_l", "requires an explicit finite nonnegative number"))
    if type(neutralize) is not bool:
        errors.append(_ion_error("neutralize", "requires an explicit boolean"))
    waters = {r for r in modeller.topology.residues() if r.name == "HOH"}
    water_ids = {id(r) for r in waters}
    partitions = []
    for name, supplied in (("retained_water_residues", retained_water_residues),
                           ("generated_water_residues", generated_water_residues)):
        if not isinstance(supplied, (list, tuple)):
            errors.append(_ion_error(name, "requires a list or tuple of live HOH residues"))
            supplied = ()
        valid = [r for r in supplied if id(r) in water_ids]
        if len(valid) != len(supplied) or len(set(valid)) != len(valid):
            errors.append(_ion_error(name, "requires unique live HOH residues; stale/foreign rows are not identity"))
        partitions.append(set(valid))
    retained, generated = partitions
    if retained & generated or retained | generated != waters:
        errors.append(_ion_error("water_partition", "retained/generated lists must disjointly cover every HOH residue"))
    neighbors = {a: [] for a in modeller.topology.atoms()}
    for a, b in modeller.topology.bonds():
        neighbors[a].append(b)
        neighbors[b].append(a)
    for residue in waters:
        atoms = list(residue.atoms())
        oxygens = [a for a in atoms if a.element == app.element.oxygen]
        hydrogens = [a for a in atoms if a.element == app.element.hydrogen]
        if (len(atoms) != 3 or len(oxygens) != 1 or len(hydrogens) != 2
                or set(neighbors[oxygens[0]]) != set(hydrogens)
                or len(neighbors[oxygens[0]]) != 2
                or any(neighbors[h] != oxygens for h in hydrogens)):
            errors.append(_ion_error(f"water[{residue.index}]",
                                     "complete isolated HOH requires one O, two H and exactly two O-H bonds"))
    _ion_raise(errors)
    before = _ion_atom_keys(modeller, unit)
    # Deep-copy the topology too: upstream failure or unsupported mutation cannot
    # damage caller-owned source objects before preservation verification.
    candidate = app.Modeller(deepcopy(modeller.topology), deepcopy(modeller.positions))
    copied_residues = list(candidate.topology.residues())
    replaceable = {}
    # Public ion sampling consumes dictionary order. Object-hashed set order
    # would make an explicit seed depend on allocation addresses, not input.
    for residue in sorted(generated, key=lambda residue: residue.index):
        copied = copied_residues[residue.index]
        oxygen = next(a for a in copied.atoms() if a.element == app.element.oxygen)
        replaceable[copied] = candidate.positions[oxygen.index]
    try:
        candidate._addIons(forcefield, len(waters), replaceable, positiveIon=positive_ion,
                           negativeIon=negative_ion, ionicStrength=ionic_strength_mol_per_l * unit.molar,
                           neutralize=neutralize)
    except Exception as error:
        raise PlanValidationErrors([_ion_error(
            "placement", f"OpenMM could not place ions using generated waters only ({error}); "
            "inspect the forcefield/structure and available generated-water sites")],
            footer="input modeller is unchanged") from error
    after = _ion_atom_keys(candidate, unit)
    previous = set(before.values())
    remaining = set(after.values())
    removed = {a for a, key in before.items() if key not in remaining}
    if any(a.residue not in generated for a in removed):
        _ion_raise([_ion_error("preservation", "ion placement removed or moved a retained atom")])
    removed_residues = {a.residue for a in removed}
    if removed != {a for r in removed_residues for a in r.atoms()}:
        _ion_raise([_ion_error("preservation", "ion placement changed only part of a generated water")])
    added = {a for a, key in after.items() if key not in previous}
    oxygen_positions = {before[a][1] for a in removed if a.element == app.element.oxygen}
    if (len(added) != len(removed_residues)
            or {after[a][1] for a in added} != oxygen_positions
            or any(a.element not in (app.element.sodium, app.element.chlorine)
                   or sum(1 for _ in a.residue.atoms()) != 1 for a in added)
            or any(a in added or b in added for a, b in candidate.topology.bonds())
            or _ion_structure(modeller, before, removed) != _ion_structure(candidate, after, added)):
        _ion_raise([_ion_error("preservation", "ion placement changed preserved atom/residue/chain/bond/box evidence")])
    modeller.topology, modeller.positions = candidate.topology, candidate.positions
