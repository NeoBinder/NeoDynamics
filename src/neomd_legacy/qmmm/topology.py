#  from openmm.app.element import Element
from copy import deepcopy

from ttk.core import element_from_symbol
from ttk import unit as ttk_unit
from ttk.io.topology_export import topology_from_indices, topology_to_openmm
from ttk.io.topology_parser import topology_from_openmmm


def compute_scale_factor_g(qm_element, mm_element, link_element):
    """
    Computes scale factor g for link atom, RC, and RCD schemes.
    The equation used to compute g is:

    .. math::
        \frac{R_{qm} + R_{link}}{R_{qm} + R_{mm}}

    where R is the pyykko covalent radius of an atom.
    Parameters
    ----------
    qm : str
        element symbol of the QM atom involved in broken bond
    mm : str
        element symbol of the MM atom involved in broken bond
    link : str
        element symbol for link atom
    Returns
    -------
    float
        g, the scaling factor
    Examples
    --------
    >>> compute_scale_factor_g('C', 'C', 'H')
    """
    r_qm = qm_element.covalent_radius_pyykko
    r_mm = mm_element.covalent_radius_pyykko
    r_link = link_element.covalent_radius_pyykko

    g = (r_qm + r_link) / (r_qm + r_mm)

    return g


def prepare_link_atoms(
    boundaries, topology, qm_top, link_atom_element="H", boundary_treatment="link_atom"
):
    links = []
    res_dict = {(res.name + res.res_seq): res for res in qm_top.get_residues()}
    atom_dict = {str(atom): atom for atom in qm_top.get_atoms()}
    for qm_atom, mm_atom, _ in boundaries:
        link = {}
        link["qm_atom"] = qm_atom
        link["mm_atom"] = mm_atom
        H_add = qm_top.add_atom(
            "H1",
            element_from_symbol(link_atom_element),
            res_dict[qm_atom.residue.name + qm_atom.residue.res_seq],
        )
        link["link_atom"] = H_add
        link["scale_factor"] = compute_scale_factor_g(
            qm_atom.element, mm_atom.element, H_add.element
        )
        g = link["scale_factor"]
        pos = (1 - g) * qm_atom.position + g * mm_atom.position
        H_add.position = pos
        qm_top.add_bond(atom_dict[str(qm_atom)], H_add, 1.0)
        if boundary_treatment in {"RC", "RCD"}:
            bonds = []
            for bond in topology.get_bonds():
                a0, a1 = bond.atom1, bond.atom2
                if a0.index == mm_atom.index and a1.index != qm_atom.index:
                    bonds.append(a1.index)
                elif a1.index == mm_atom.index and a0.index != qm_atom.index:
                    bonds.append(a0.index)
            link["bonds_to_mm"] = bonds
        links.append(link)
    return links


def find_boundaries(topology, qm_indices):
    qm_indices = set(qm_indices)
    boundaries = []
    atoms = [atom for atom in topology.get_atoms() if atom.index in qm_indices]
    for atom in atoms:
        for bond in atom.bonds:
            a1, a2 = bond.atom1, bond.atom2
            index_set = set([a1.index, a2.index])
            if len(index_set.difference(qm_indices)) == 1:
                if a1.index in qm_indices:
                    qm_atom, mm_atom = a1, a2
                else:
                    qm_atom, mm_atom = a2, a1
                boundaries.append([qm_atom, mm_atom, bond])
    return boundaries


class QMTopology:

    def __init__(
        self, complex_topology, qm_topology, links, qm_indices, boundary_treatment
    ):
        self.complex_topology = complex_topology
        self.qm_topology = qm_topology
        self.links = links
        self.qm_indices = qm_indices
        self.boundary_treatment = boundary_treatment
        self.link_atoms_indices = [link["link_atom"].index for link in self.links]

    @classmethod
    def from_neosystem(
        cls, neosystem, qm_indices, boundary_treatment="link_atom", start_idx=1
    ):
        model = topology_from_openmmm(
            neosystem.topology, neosystem.positions, keepIdx=True
        )
        boundaries = find_boundaries(model, qm_indices)
        qm_topology = topology_from_indices(model, qm_indices, start_idx)
        if len(boundaries):
            qmmmlinks = prepare_link_atoms(
                boundaries,
                model,
                qm_topology,
                boundary_treatment=boundary_treatment,
            )
        else:
            qmmmlinks = []
        qm_topology.create_index()
        qm_region = cls(model, qm_topology, qmmmlinks, qm_indices, boundary_treatment)
        return qm_region

    def update_pos(self, complex_positions):
        for link in self.links:
            qm_atom = link["qm_atom"]
            mm_atom = link["mm_atom"]
            H_add = link["link_atom"]
            g = link["scale_factor"]
            # find link_atom: H_add position
            pos = (1 - g) * complex_positions[qm_atom.index] + g * complex_positions[
                mm_atom.index
            ]
            H_add.position = pos * ttk_unit.nanometer
        for atom in self.qm_topology.get_atoms():
            if atom.name != "H1":
                atom.position = (
                    complex_positions[atom.properties["origin_index"]]
                    * ttk_unit.nanometer
                )
        return self.qm_topology.get_positions()

    def get_xyz_format_geometry(self):
        geometry = ""
        positions = self.qm_topology.get_positions().to(ttk_unit.angstrom).magnitude
        symbol_ls = [atom.element.symbol for atom in self.qm_topology.get_atoms()]

        for index in range(len(symbol_ls)):
            symbol = symbol_ls[index]
            pos = positions[index]
            geometry += "{} {: > 7.3f} {: > 7.3f} {: > 7.3f} \n".format(
                symbol, *pos.tolist()
            )
        geometry = "".join(geometry)
        return geometry

    def topology_to_openmm(self):
        return topology_to_openmm(self.qm_topology)
