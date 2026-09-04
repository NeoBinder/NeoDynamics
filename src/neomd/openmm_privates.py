"""openmm_privates — the ONE isolated home for openmm private-API usage.

The system-preparation workflow needs three openmm internals that have no
public equivalent (as of the pinned release):

* ``Topology._standardBonds`` / ``Topology._bonds`` + the
  ``createStandardBonds``/``createDisulfideBonds`` rebuild — teaching a
  topology about custom residues' bond graph (:func:`custom_bonds`);
* ``Modeller._ResidueData`` / ``Modeller._residueHydrogens`` /
  ``Modeller._Hydrogen`` — teaching ``modeller.addHydrogens`` about custom
  residues' hydrogen parents (:func:`custom_addH`); reads
  ``ForceField._atomTypes`` for atom elements;
* ``openmm.app.internal.unitcell.computePeriodicBoxVectors`` — the
  ligand-extent fallback box (:func:`compute_periodic_box_vectors`).

Everything private lives in THIS file and nowhere else (enforced by
tests/v2/test_system.py's source scan).  The usages are PINNED: the first
call checks ``openmm.__version__`` against :data:`PINNED_OPENMM_PREFIXES`
and raises :class:`~neomd.errors.UpstreamVersionError` outside the range —
bumping openmm past the pin produces a loud refusal, never silent
behavioral drift.  Re-pin by verifying the three usages against the new
release (the smoke tests in test_system.py cover each one) and extending
the prefix tuple.
"""

from __future__ import annotations

import xml.etree.ElementTree as etree

from .errors import UpstreamVersionError

__all__ = [
    "PINNED_OPENMM_PREFIXES",
    "assert_pinned_openmm",
    "custom_bonds",
    "custom_addH",
    "compute_periodic_box_vectors",
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
