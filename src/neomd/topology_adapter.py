"""OpenMM views of authoritative molecular-topology records.

OpenMM's structure readers and ``createStandardBonds`` are deliberately absent:
only supplied Bond rows enter its chemical topology.  Synthetic containers needed
by OpenMM for parentless entities are private to this view; ``records`` remains
the lossless source of identity, metadata and membership.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .topology_records import TopologyRecordError, validate_topology


class ConnectivityPrerequisiteError(ValueError):
    """An operation needs a topology-wide complete connectivity assertion."""


@dataclass(frozen=True)
class ScientificTopologyView:
    records: dict
    topology: object
    positions_nm: tuple
    atom_by_id: object
    index_by_id: object
    frame_id: str

    @property
    def coverage(self):
        return self.records["coverage"]

    def require_complete(self, operation: str):
        if self.coverage["state"] != "complete":
            raise ConnectivityPrerequisiteError(
                f"{operation} requires topology-wide complete explicit connectivity; "
                f"received {self.coverage['state']} coverage"
            )


def openmm_view(records, *, frame_id=None):
    """Map every supplied atom and bond once, retaining the original flat tables.

    Coordinate rows are joined by stable atom ID, never row position. The first
    frame is selected only when the caller omits a frame ID. Distinct model frames
    cannot be combined by atom name or distance.
    """
    from openmm import Vec3
    from openmm.app import Topology, element
    from openmm.unit import nanometer

    validate_topology(records)
    frames = {frame["id"]: frame for frame in records["frames"]}
    selected = frame_id or records["frames"][0]["id"]
    if selected not in frames:
        raise TopologyRecordError("unknown coordinate frame ID")
    frame = frames[selected]
    scale = 1.0 if frame["unit"] == "nm" else 0.1
    topology = Topology()
    chains = {row["id"]: topology.addChain(row["id"]) for row in records["chains"]}
    # An OpenMM atom always has a residue and chain. These containers do not
    # change the canonical membership rows or imply chemical edges.
    orphan_chain = None
    residues = {}
    for row in records["residues"]:
        chain = chains.get(row["chain_id"])
        if chain is None:
            if orphan_chain is None:
                orphan_chain = topology.addChain("__unassigned__")
            chain = orphan_chain
        residues[row["id"]] = topology.addResidue(row["name"], chain, row["id"])
    orphan_residue = None
    atoms = {}
    for row in records["atoms"]:
        residue = residues.get(row["residue_id"])
        if residue is None:
            if orphan_chain is None:
                orphan_chain = topology.addChain("__unassigned__")
            if orphan_residue is None:
                orphan_residue = topology.addResidue("UNASSIGNED", orphan_chain, "__unassigned__")
            residue = orphan_residue
        symbol = row["element"]
        try:
            chemical_element = element.get_by_symbol(symbol) if symbol is not None else None
        except KeyError as error:
            raise TopologyRecordError(f"unsupported element for atom {row['id']}") from error
        atoms[row["id"]] = topology.addAtom(
            row["name"], chemical_element, residue, row["id"],
            formalCharge=row["formal_charge"],
        )
    order_values = {"single": 1, "double": 2, "triple": 3}
    for row in records["bonds"]:
        order = row["order"]["kind"]
        # OpenMM cannot express an arbitrary named order. Keep it in records and
        # leave the OpenMM annotation unknown, without changing the edge.
        topology.addBond(atoms[row["atom_ids"][0]], atoms[row["atom_ids"][1]],
                         order=order_values.get(order))
    positions = {row["atom_id"]: row["xyz"] for row in frame["positions"]}
    coordinates = tuple(Vec3(*(value * scale for value in positions[row["id"]]))
                        for row in records["atoms"])
    if frame["box"] is not None:
        topology.setPeriodicBoxVectors(tuple(Vec3(*(value * scale for value in vector))
                                             for vector in frame["box"]) * nanometer)
    return ScientificTopologyView(
        records, topology, coordinates, MappingProxyType(atoms),
        MappingProxyType({key: atom.index for key, atom in atoms.items()}), selected,
    )


def parameterized_system(view, *, masses_amu, bond_parameters):
    """Create a narrow real OpenMM System using only caller-supplied parameters.

    This is an explicit harmonic-bond intake path, not protein force-field
    preparation. It has no implicit masses, radii, charges, angles or bond terms.
    A caller seeking protein ff14SB preparation must use a separately qualified
    operation; these parameters cannot claim that qualification.
    """
    import math

    import openmm

    view.require_complete("parameterized system construction")
    atom_ids = {row["id"] for row in view.records["atoms"]}
    bond_pairs = {tuple(row["atom_ids"]) for row in view.records["bonds"]}
    if set(masses_amu) != atom_ids or set(bond_parameters) != bond_pairs:
        raise TopologyRecordError("explicit parameters must cover every atom and bond exactly")
    system = openmm.System()
    for row in view.records["atoms"]:
        mass = masses_amu[row["id"]]
        if type(mass) not in (int, float) or not math.isfinite(mass) or mass <= 0:
            raise TopologyRecordError("atom mass must be positive and finite")
        system.addParticle(mass)
    force = openmm.HarmonicBondForce()
    for row in view.records["bonds"]:
        pair = tuple(row["atom_ids"])
        length_nm, stiffness_kj_mol_nm2 = bond_parameters[pair]
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in
                (length_nm, stiffness_kj_mol_nm2))
                or length_nm <= 0 or stiffness_kj_mol_nm2 < 0):
            raise TopologyRecordError("bond parameters must be finite with positive length")
        force.addBond(view.index_by_id[pair[0]], view.index_by_id[pair[1]],
                      length_nm, stiffness_kj_mol_nm2)
    system.addForce(force)
    return system
