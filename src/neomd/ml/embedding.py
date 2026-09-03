"""embedding — mechanical embedding, ported VERBATIM from openmm-ml (ADR-0004).

Source: openmm-ml 1.7, commit 501c3a04062db52c76a7bd0b67e6bc62a2e48ae3
(2026-08-29), files ``openmmml/embeddings/mechanicalembedding.py`` and
``openmmml/embeddings/utilities.py`` — MIT license, copyright (c) 2026
Stanford University; authors Peter Eastman, Evan Pretti.  The MIT permission
notice below applies to the ported code (settlement decision #2: physics is
ported verbatim, with attribution, never "cleaned up").

What mechanical embedding does (unchanged from the source): the conventional
force field keeps computing the ML↔MM interactions (the ML atoms' MM point
charges are NOT zeroed — they still carry the ML↔MM electrostatics), while
every MM term BETWEEN ML atoms is removed so the ML potential owns the ML
region's internal energy:

* bonded terms whose atoms are ALL in the ML region are dropped
  (:func:`removeBonds`, XML round-trip, bonds/angles/torsions);
* NonbondedForce gets zero-valued exceptions (charge product 0, sigma 1 nm,
  epsilon 0) for every ML-ML pair — unless the ML model itself computes
  long-range electrostatics (``ml_long_range``), in which case the true
  charge products are KEPT and the ML-ML PME contribution is subtracted
  through a separate PME NonbondedForce holding only the ML charges
  (wrapped in ``-excludeForce`` CustomCVForce);
* CustomNonbondedForce objects get ML-ML exclusions.

Documented deviations from the source (each is a seam, not physics):

1. the source's two ``MLPotentialImpl`` seams become plain parameters:
   ``potential.getMLLongRange()`` -> the ``ml_long_range`` argument (our
   TorchScript models cannot be probed, so the plan declares it),
   ``potential.addForces(...)`` -> the ``add_ml_forces`` callback (the
   caller installs its NNP force — mock or TorchForce — into the returned
   system);
2. the interpolation path (``InterpolationHelper``) is not ported — neomd
   has no interpolation use case; only the ``interpolate=False`` branch is;
3. the source calls ``utilities.makeCustomNonbondedExclusions`` while
   utilities.py defines ``addCustomNonbondedExclusions`` — a latent upstream
   naming bug (AttributeError on any system carrying a CustomNonbondedForce);
   this port calls the DEFINED name.

Everything else — the periodicity logic, the exception parameters, the PME
exclusion-force construction, the error messages — is the source verbatim.

MIT license of the ported code:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

from typing import Callable

__all__ = ["removeBonds", "addCustomNonbondedExclusions", "create_mixed_system"]


# ---------------------------------------------------------------------------
# utilities.py: removeBonds (verbatim)
# ---------------------------------------------------------------------------


def removeBonds(system, atoms: list[int], removeInSet: bool):
    """
    Copy a System, removing all bonded interactions between atoms in (or not in)
    a particular set.

    Parameters
    ----------
    system: System
        The System to copy.
    atoms: list[int]
        A set of atom indices.
    removeInSet: bool
        If True, any bonded term connecting atoms in the specified set is
        removed.  If False, any term that does *not* connect atoms in the
        specified set is removed.

    Returns
    -------
    A newly created System object in which the specified bonded interactions
    have been removed.
    """
    import xml.etree.ElementTree as ET

    import openmm

    atomSet = set(atoms)

    # Create an XML representation of the System.

    xml = openmm.XmlSerializer.serialize(system)
    root = ET.fromstring(xml)

    # This function decides whether a bonded interaction should be removed.

    def shouldRemove(termAtoms):
        return all(a in atomSet for a in termAtoms) == removeInSet

    # Remove bonds, angles, and torsions.

    for bonds in root.findall('./Forces/Force/Bonds'):
        for bond in bonds.findall('Bond'):
            bondAtoms = [int(bond.attrib[p]) for p in ('p1', 'p2')]
            if shouldRemove(bondAtoms):
                bonds.remove(bond)
    for angles in root.findall('./Forces/Force/Angles'):
        for angle in angles.findall('Angle'):
            angleAtoms = [int(angle.attrib[p]) for p in ('p1', 'p2', 'p3')]
            if shouldRemove(angleAtoms):
                angles.remove(angle)
    for torsions in root.findall('./Forces/Force/Torsions'):
        for torsion in torsions.findall('Torsion'):
            torsionLabels = ('p1', 'p2', 'p3', 'p4') if 'p1' in torsion.attrib \
                else ('a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4')
            torsionAtoms = [int(torsion.attrib[p]) for p in torsionLabels]
            if shouldRemove(torsionAtoms):
                torsions.remove(torsion)

    # Create a new System from it.

    return openmm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))


# ---------------------------------------------------------------------------
# utilities.py: addCustomNonbondedExclusions (verbatim; the source's caller
# misspells this name — see the module docstring, deviation 3)
# ---------------------------------------------------------------------------


def addCustomNonbondedExclusions(force, atoms: list[int]) -> None:
    """
    Adds exclusions between all atoms in a set to a CustomNonbondedForce.

    Parameters
    ----------
    force: openmm.CustomNonbondedForce
        The CustomNonbondedForce to modify in place.
    atoms: list[int]
        A set of atom indices.
    """

    # Only call addExclusion for those particle pairs not already excluded.

    existing = set(tuple(force.getExclusionParticles(i))
                   for i in range(force.getNumExclusions()))
    for iAtom1 in range(len(atoms)):
        atom1 = atoms[iAtom1]
        for iAtom2 in range(iAtom1):
            atom2 = atoms[iAtom2]
            if (atom1, atom2) not in existing and (atom2, atom1) not in existing:
                force.addExclusion(atom1, atom2)


# ---------------------------------------------------------------------------
# mechanicalembedding.py: MechanicalEmbedding.createMixedSystem
# (the interpolate=False branch, verbatim; the seams adapted per the module
# docstring)
# ---------------------------------------------------------------------------


def create_mixed_system(system, atoms: list[int], ml_long_range,
                        add_ml_forces: Callable[[object], None]):
    """Mechanical embedding (openmm-ml ``MechanicalEmbedding`` port).

    Parameters
    ----------
    system: openmm.System
        The pure-MM System (freshly deserialized; must be XML-serializable —
        no NNP force may live in it yet, which is why assembly happens BEFORE
        the ML force is added, inside the openmm adapter, pre-Context).
    atoms: list[int]
        The ML region's 0-based particle indices.
    ml_long_range: bool | None
        Whether the ML potential computes long-range (Ewald/PME)
        electrostatics: ``None`` = undeclared (periodic systems refuse, the
        source's behavior), ``False``/``True`` = the plan's
        ``long_range_electrostatics`` declaration.
    add_ml_forces: Callable[[openmm.System], None]
        The NNP installation callback (mock or TorchScript): called with the
        mixed System AFTER the embedding is applied; owns adding its Force(s)
        and force-group assignment.

    Returns
    -------
    A NEW openmm.System (the XML round-trip in ``removeBonds`` copies it).
    """
    import openmm

    periodic = system.usesPeriodicBoundaryConditions()

    # See if the MM force field uses long-range interactions.

    mmLongRange = False
    mmLongRangeForce = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if force.getNonbondedMethod() in (openmm.NonbondedForce.Ewald,
                                              openmm.NonbondedForce.PME,
                                              openmm.NonbondedForce.LJPME):
                mmLongRange = True
                if mmLongRangeForce is not None:
                    raise ValueError(
                        "Multiple long-range NonbondedForce objects encountered.")
                mmLongRangeForce = force
                break

    excludeLongRange = False

    if periodic:

        # For a non-periodic system, we will always use exceptions for the
        # ML subset.  Otherwise, we need to know whether or not the ML
        # potential is long-range.

        if ml_long_range is None:
            raise ValueError(
                "The system is periodic and it is unknown if the ML model uses "
                "long-range interactions; provide the long_range_electrostatics "
                "option (ml_region.model) to specify.")

        # We don't support the case where the MM force field is not
        # long-range but the ML potential is.

        if ml_long_range:
            if not mmLongRange:
                raise ValueError(
                    "The system is periodic and the ML model uses long-range "
                    "interactions while the MM force field does not.")

            excludeLongRange = True

    # Create the new system with ML-ML interactions to be computed by the ML
    # potential removed.

    newSystem = removeBonds(system, atoms, True)

    for force in newSystem.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # All of the LJ interactions in the ML region should be zeroed.
            charges = [force.getParticleParameters(atom)[0] for atom in atoms]
            for iAtom1 in range(len(atoms)):
                for iAtom2 in range(iAtom1):

                    # If the ML region electrostatics should be excluded in
                    # long-range, keep this exception's charge product set
                    # to the true product of the charges since the energy
                    # for this pair will be subtracted later.  If not, set
                    # it to zero here.

                    if excludeLongRange:
                        chargeProd = charges[iAtom1] * charges[iAtom2]
                    else:
                        chargeProd = 0
                    force.addException(atoms[iAtom1], atoms[iAtom2],
                                       chargeProd, 1, 0, True)

            # This may cause exceptions in the MM region to use PBCs, but
            # this should not ordinarily have any significant effects.

            force.setExceptionsUsePeriodicBoundaryConditions(periodic)

        elif isinstance(force, openmm.CustomNonbondedForce):
            addCustomNonbondedExclusions(force, atoms)

    if excludeLongRange:
        # Prepare a force to calculate the PME energy of the ML-ML region.

        excludeForce = openmm.NonbondedForce()
        excludeForce.setCutoffDistance(mmLongRangeForce.getCutoffDistance())
        excludeForce.setEwaldErrorTolerance(
            mmLongRangeForce.getEwaldErrorTolerance())
        excludeForce.setNonbondedMethod(openmm.NonbondedForce.PME)
        excludeForce.setPMEParameters(*mmLongRangeForce.getPMEParameters())

        atomSet = set(atoms)
        for atom in range(newSystem.getNumParticles()):
            excludeForce.addParticle(
                mmLongRangeForce.getParticleParameters(atom)[0]
                if atom in atomSet else 0, 1, 0)

        # Add the ML potential and subtract the ML-ML PME energy if needed.

        cvForce = openmm.CustomCVForce("-excludeForce")
        cvForce.addCollectiveVariable("excludeForce", excludeForce)
        newSystem.addForce(cvForce)

    add_ml_forces(newSystem)

    return newSystem
