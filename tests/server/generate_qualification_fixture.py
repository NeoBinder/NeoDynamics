"""Regenerate the tiny public-API fixture; audit hashes after an engine upgrade."""

from pathlib import Path

import numpy as np
import openmm
from openmm import app, unit

from neomd.system import SystemBundle, export_system_bundle


def generate(destination):
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        topology = app.Topology()
        chain = topology.addChain("A")
        atoms = [topology.addAtom(f"AR{i}", app.element.argon,
                                 topology.addResidue("ARG", chain, str(i + 1))) for i in range(6)]
        positions = np.asarray([[0, 0, 0], [.3, 0, 0], [.3, .3, 0],
                                [.3, .3, .3], [.6, .2, .2], [.7, .4, .1]])
        system = openmm.System()
        harmonic = openmm.CustomBondForce("lambda_sterics*0.5*k*(r-r0)^2")
        harmonic.addGlobalParameter("lambda_sterics", 1)
        harmonic.addPerBondParameter("k")
        harmonic.addPerBondParameter("r0")
        for _ in atoms:
            system.addParticle(39.9)
        for i in range(5):
            topology.addBond(atoms[i], atoms[i + 1])
            harmonic.addBond(i, i + 1, [100, float(np.linalg.norm(positions[i + 1] - positions[i]))])
        system.addForce(harmonic)
        torsion = openmm.PeriodicTorsionForce()
        torsion.addTorsion(0, 1, 2, 3, 1, 0, 0.1)
        system.addForce(torsion)
        source = root / "source.pdbx"
        with source.open("w") as handle:
            app.PDBxFile.writeFile(topology, positions * unit.nanometer, handle)
        xml = root / "system.xml"
        xml.write_text(openmm.XmlSerializer.serialize(system))
        bundle = export_system_bundle(SystemBundle(topology_file=str(source), system_xml=str(xml)), root / "export")
        shutil.copytree(Path(bundle.prepared_manifest).parent, destination)


if __name__ == "__main__":
    generate(Path(__file__).parents[2] / "src/neomd_server/fixtures/qualification")
