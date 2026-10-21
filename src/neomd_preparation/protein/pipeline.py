"""Atomic local canonical-protein preparation through a verified System bundle.

No service endpoint, runtime protocol, image or deployment policy lives here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from neomd_preparation.bundle import _error, _raise


def prepare_protein_system(source, directory, *, parameters, pH, padding_nm,
                           ionic_strength_mol_per_l, platform, seed):
    """Group, repair, hydrogenate, solvate and assemble one canonical ProteinSource.

    All scientific settings are explicit, including the eleven accepted System
    settings. Existing atoms (including H and source water) never move or vanish.
    A new immutable UUID directory becomes visible only after full independent
    history and System/report verification. Failure preserves caller data and
    removes only this invocation's private staging directory. This is local
    atomic publication, not a power-loss durability or scientific stability claim.
    """
    from neomd.system import SystemBundle
    from neomd_preparation.assembly.report import assemble_protein_system
    from neomd_preparation.assembly.system import _validated_parameters
    from neomd_preparation.protein.hydrogens import add_protein_hydrogens
    from neomd_preparation.protein.repair import group_and_repair_protein
    from neomd_preparation.protein.solvent import _settings, solvate_protein
    from neomd_preparation.verify.pipeline import verify_protein_preparation

    _validated_parameters(parameters)
    _settings(padding_nm, ionic_strength_mol_per_l, seed)
    repaired = group_and_repair_protein(source, platform=platform, seed=seed)
    hydrogenated = add_protein_hydrogens(repaired, pH=pH, platform=platform, seed=seed)
    solvated = solvate_protein(hydrogenated, padding_nm=padding_nm,
                               ionic_strength_mol_per_l=ionic_strength_mol_per_l, seed=seed)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    final = root / solvated.lineage.revision_id
    if final.exists():
        _raise([_error("prepared_revision_id", "immutable revision already exists")])
    with tempfile.TemporaryDirectory(prefix=".preparation-", dir=root) as temporary:
        bundle = assemble_protein_system(
            solvated.topology, solvated.positions_nm, temporary,
            lineage=solvated.lineage, parameters=parameters)
        verify_protein_preparation(bundle.prepared_manifest)
        os.rename(Path(bundle.prepared_manifest).parent, final)
    return SystemBundle(topology_file=str(final / "topology.pdbx"),
                        system_xml=str(final / "system.xml"),
                        prepared_manifest=str(final / "manifest.json"))
