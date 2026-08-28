"""Regenerate the gas-phase alanine dipeptide (ACE-ALA-NME) micro-fixture.

Run:  pixi run -e test python tests/data/ala2/make_fixture.py

Outputs (overwrite in place, deterministic on the CPU platform):
  - ala2.pdb    : 22-atom capped alanine dipeptide, energy-minimized coordinates
  - system.xml  : serialized OpenMM System (amber14, vacuum, NoCutoff, HBonds constraints)

The starting heavy-atom/hydrogen layout is the classic PLUMED ala2 structure;
it is minimized here so the committed fixture is mechanically sane.  Atom order
(and therefore the phi/psi dihedral indices 4-6-8-14 / 6-8-14-16, 0-based)
follows that layout.
"""
import os

import openmm
from openmm import app, unit

HERE = os.path.dirname(os.path.abspath(__file__))

# Classic alanine dipeptide coordinates (ACE-ALA-NME), PLUMED masterclass ala2.
ATOMS = [
    # (name, resName, resId, x, y, z)
    ("HH31", "ACE", 1, -1.280, -0.013, 0.000),
    ("CH3", "ACE", 1, -1.295, 1.040, 0.000),
    ("HH32", "ACE", 1, -2.121, 1.287, 0.663),
    ("HH33", "ACE", 1, -0.477, 1.434, 0.437),
    ("C", "ACE", 1, -1.467, 1.723, -1.240),
    ("O", "ACE", 1, -1.406, 1.216, -2.340),
    ("N", "ALA", 2, -1.553, 3.059, -1.223),
    ("H", "ALA", 2, -1.584, 3.601, -0.413),
    ("CA", "ALA", 2, -1.685, 3.700, -2.501),
    ("HA", "ALA", 2, -0.959, 3.278, -3.183),
    ("CB", "ALA", 2, -1.364, 5.179, -2.497),
    ("HB1", "ALA", 2, -1.459, 5.566, -1.484),
    ("HB2", "ALA", 2, -1.975, 5.772, -3.173),
    ("HB3", "ALA", 2, -0.338, 5.368, -2.809),
    ("C", "ALA", 2, -3.076, 3.411, -2.931),
    ("O", "ALA", 2, -3.490, 2.270, -2.993),
    ("N", "NME", 3, -3.855, 4.416, -3.219),
    ("H", "NME", 3, -3.571, 5.372, -3.084),
    ("CH3", "NME", 3, -5.251, 4.187, -3.592),
    ("HH31", "NME", 3, -5.434, 3.599, -4.484),
    ("HH32", "NME", 3, -5.630, 3.599, -2.718),
    ("HH33", "NME", 3, -5.842, 5.122, -3.731),
]


def build_pdb():
    lines = []
    for i, (name, res, resid, x, y, z) in enumerate(ATOMS, start=1):
        element = "H" if name.startswith("H") else name[0]
        lines.append(
            "ATOM  {:>5} {:>4} {:>3} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}"
            "  1.00  0.00          {:>2}".format(i, name, res, resid, x, y, z, element)
        )
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def main():
    pdb_path = os.path.join(HERE, "ala2.pdb")
    with open(pdb_path, "w") as f:
        f.write(build_pdb())

    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amber14-all.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=False,
    )
    integrator = openmm.LangevinIntegrator(298 * unit.kelvin, 1 / unit.picosecond,
                                           0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(4242)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=5000)
    e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print("atoms:", system.getNumParticles())
    print("energy before/after minimization: {} -> {}".format(e0, e1))

    positions = simulation.context.getState(getPositions=True).getPositions()
    with open(pdb_path, "w") as f:
        app.PDBFile.writeFile(pdb.topology, positions, f, keepIds=True)
    with open(os.path.join(HERE, "system.xml"), "w") as f:
        f.write(openmm.XmlSerializer.serialize(system))
    print("wrote", pdb_path, "and system.xml")


if __name__ == "__main__":
    main()
